import copy
import itertools

import torch
import torch.nn.functional as F
import numpy as np
import wandb

from decay import EpsilonDecay

from replay_buffer import PrioritizedReplayBuffer, FrameStacker
from tracking import Tracker
from elo_system import HockeyTournamentEvaluation
import crps
from agent import NNAgent
import plotting

TRAINING_SCHEDULES = ['basic', 'adv1', 'adv2', 'self-play', 'dynamic'] 

class DqnInferenceAgent(NNAgent):

    def __init__(self, model, device, frame_stacks=None, loss=None, phi=None, softactions=None):
        super().__init__(model, device, frame_stacks)
        self.loss = loss if loss is not None else 'td'
        self.phi = phi if phi is not None else 0.0
        self.softactions = softactions if softactions is not None else False


    @classmethod
    def _load_model(cls, model, config, device):
        return cls(
            model, 
            device, 
            frame_stacks=config['frame_stacks'], 
            loss=config.get('loss'), 
            phi=config.get('phi'),
            softactions=config.get('softactions')
        )


    def select_action(self, state, frame_idx=None, train=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if self.loss == 'ndqn':
            combined = self.model(state)
            mean, std = combined.chunk(2, dim=1)
            std = crps.positive_std(std)
            Qs = mean + self.phi*std
        if self.loss == 'crps':
            combined = self.model(state)
            mean, std, _, _ = combined.chunk(4, dim=1)
            std = crps.positive_std(std)
            Qs = mean + self.phi*std
        elif self.loss == 'td':
            Qs = self.model(state)
        else:
            raise ValueError(f"Unknown loss: {self.loss}")

        if not self.softactions:
            return Qs.argmax(dim=1).item()
        else:
            probs = F.softmax(Qs, dim=1).squeeze(0)
            return np.random.choice(len(probs), p=probs.cpu().detach().numpy())

    def copy(self, eval=True):
        model = copy.deepcopy(self.model)
        model.requires_grad_(False)
        if eval:
            model.eval().requires_grad_(False)
        return DqnInferenceAgent(
            model,
            self.device,
            frame_stacks=self.stacker.num_frames,
            loss=self.loss,
            phi=self.phi,
            softactions=self.softactions
        )

class DqnAgent(DqnInferenceAgent):

    def __init__(self, model, optimizer, num_actions, device, frame_stacks=None, epsilon_decay=None, gamma=None, target_update_frequency=None, double_q=False, scheduler=None, softactions=None, loss=None, phi=None):
        super().__init__(model, device, frame_stacks, softactions=softactions, loss=loss, phi=phi)
        self.target_model = copy.deepcopy(model)
        self.optimizer = optimizer
        self.num_actions = num_actions
        self.epsilon_decay = epsilon_decay if epsilon_decay is not None else EpsilonDecay(constant_eps=0.0)
        self.gamma = gamma if gamma is not None else 0.99
        self.target_update_frequency = target_update_frequency if target_update_frequency is not None else 1000
        self.no_double = not double_q if double_q is not None else True
        self.scheduler = scheduler
        self.loss = loss

        self.num_updates = 0
        self.target_model.requires_grad_(False)
        self.target_model.eval()

    def select_action(self, state, frame_idx=None, train=False):
        if not train:
            return super().select_action(state, frame_idx, train=False)

        epsilon = 0.0 if frame_idx is None else self.epsilon_decay(frame_idx)
        if epsilon > 0.0 and epsilon > np.random.random():
            action = np.random.randint(self.num_actions)
        else:
            action = super().select_action(state, frame_idx, train=True)

        return action

    def update_model(self, samples, frame_idx=None):
        elementwise_loss = self.compute_loss(samples)
        weights = samples["weights"] # PER weights
        loss = torch.mean(elementwise_loss * weights)

        # Update Network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        
        self.num_updates += 1
        if self.num_updates % self.target_update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item(), elementwise_loss.detach()


    def _td_loss(self, state, action, reward, next_state, done):
        action = action.unsqueeze(1)  # B x 1
        
        next_q_value = self.target_model(next_state)
        if self.no_double:
            best_action = next_q_value.argmax(dim=1, keepdim=True) # B x 1
        else:
            best_action = self.model(next_state).argmax(dim=1, keepdim=True) # B x 1
        
        next_value_f = next_q_value.gather(1, best_action) # B x 1
        next_value_f = next_value_f.squeeze(1).detach() # B
        
        mask = 1 - done # B
        target = (reward + self.gamma * next_value_f * mask)

        cur_q_value = self.model(state).gather(1, action) # B x 1
        cur_q_value = cur_q_value.squeeze(1) # B
        loss = F.smooth_l1_loss(cur_q_value, target, reduction="none")
        return loss

    def _ndqn_loss(self, state, action, reward, next_state, done):
        action = action.unsqueeze(1)  # B x 1
        
        next_q_value, next_std = self.target_model(next_state).chunk(2, dim=1) # B x D
        next_std = crps.positive_std(next_std)
        if self.no_double:
            best_action = next_q_value.argmax(dim=1, keepdim=True) # B x 1
        else:
            next_q_prime = self.model(next_state).chunk(2, dim=1)[0] # B x D
            best_action = next_q_prime.argmax(dim=1, keepdim=True) # B x 1
        
        next_val_dist_mean = next_q_value.gather(1, best_action).squeeze(1) 
        next_val_dist_std = next_std.gather(1, best_action).squeeze(1)
        next_val_dist_std = crps.positive_std(next_val_dist_std)

        mu_1 = self.gamma * next_val_dist_mean + reward # B
        sigma_1 = self.gamma * next_val_dist_std

        # terminal states:
        done = done.bool()
        mu_1[done] = reward[done]
        sigma_1[done] = 0.05

        cur_q_value, cur_std = self.model(state).chunk(2, dim=1) # B x D
        cur_std = crps.positive_std(cur_std)
        mu_2 = cur_q_value.gather(1, action).squeeze(1) # B
        sigma_2 = cur_std.gather(1, action).squeeze(1) # B
        loss =  crps.normal_kl_div(mu_1.detach(), sigma_1.detach(), mu_2, sigma_2)
        return loss


    def _crps_loss(self, state, action, reward, next_state, done):
        action = action.unsqueeze(1)  # B x 1
        
        # CRPS for reward
        mu_cur, std_cur, mu_reward, std_reward = self.model(state).chunk(4, dim=1) # B x D
        mu_cur = mu_cur.gather(1, action).squeeze(1) # B
        std_cur = crps.positive_std(std_cur.gather(1, action).squeeze(1)) # B
        mu_reward  = mu_reward.gather(1, action).squeeze(1) # B
        std_reward = crps.positive_std(std_reward.gather(1, action).squeeze(1)) # B
        reward_loss = crps.crps_loss(reward, mu_reward, std_reward)

        # KL-div to train Q
        mu_next, std_next, _, _ = self.target_model(next_state).chunk(4, dim=1) # B x D
        std_next = crps.positive_std(std_next)
        best_action = mu_next.argmax(dim=1, keepdim=True) # B x 1
        mu_next_value, std_next_value = mu_next.gather(1, best_action).squeeze(1), std_next.gather(1, best_action).squeeze(1) # B

        _,_, mu_reward_target, std_reward_target  = self.target_model(state).chunk(4, dim=1) # B x D
        std_reward_target = crps.positive_std(std_reward_target)
        mu_reward_target = mu_reward_target.gather(1, action).squeeze(1) # B
        std_reward_target = std_reward_target.gather(1, action).squeeze(1) # B

        target_mu = mu_reward_target + (1-done) * self.gamma * mu_next_value
        target_std = std_reward_target + (1-done) * self.gamma * std_next_value
        q_loss = crps.normal_kl_div(target_mu.detach(), target_std.detach(), mu_cur, std_cur)
        
        return reward_loss + q_loss

    def compute_loss(self, samples):
        state = samples['obs']
        next_state = samples['next_obs']
        action = samples['acts']
        reward = samples['rews']
        done = samples['done']

        if self.loss == 'td':
            return self._td_loss(state, action, reward, next_state, done)
        elif self.loss == 'ndqn':
            return self._ndqn_loss(state, action, reward, next_state, done)
        elif self.loss == 'crps':
            return self._crps_loss(state, action, reward, next_state, done)
        else:
            raise ValueError(f"Unknown loss: {self.loss}")


class DqnTrainer:

    def __init__(self, model_dir, env, agent, replay_buffer, device, frame_stacks=1, training_delay=100_000, update_frequency=1, checkpoint_frequency=100_000, schedule=None):
        assert schedule is None or schedule in TRAINING_SCHEDULES
        
        self.model_dir = model_dir
        self.env = env
        self.agent = agent
        self.device = device
        self.replay_buffer = replay_buffer
        self.training_delay = training_delay
        self.update_frequency = update_frequency
        self.checkpoint_frequency = checkpoint_frequency
        self.schedule = schedule

        self.stacker = FrameStacker(frame_stacks)        
        self.tracker = Tracker()
        self.last_update = 0

        self.tournament = HockeyTournamentEvaluation()
        self.tournament.add_agent('self', self.agent)
        self.tournament.add_agent('lilith_weak', NNAgent.load_lilith_weak(self.device))
        
    def reset_env(self):
        self.stacker.clear()
        state, info = self.env.reset()
        return self.stacker.append_and_stack(state)

    def step(self, action):
        next_state, reward, done, _, info = self.env.step(action)
        next_state = self.stacker.append_and_stack(next_state)
        return next_state, reward, done, info

    def prepopulate(self, agent, num_frames: int):
        print('Prepopulating replay buffer...')
        self.stacker.clear()
        state, _ = self.env.reset()
        self.env.add_opponent('TEMP-AGENT', agent.copy(), prob=4)
        state_self = self.stacker.append_and_stack(state)
        for frame_idx in range(1, num_frames + 1):
            action = agent.select_action(state, train=False)
            next_state, reward, done, _, info = self.env.step(action)
            next_state_self = self.stacker.append_and_stack(next_state)
            self.replay_buffer.store(state_self, action, reward, next_state_self, done)
            state = next_state
            if done:
                self.stacker.clear()
                state, _ = self.env.reset()
                state_self = self.stacker.append_and_stack(state)
        self.env.remove_opponent('TEMP-AGENT')

    def train(self, num_frames: int):
        # Warmup
        print('Warming up...')
        state = self.reset_env()
        for idx in range(self.training_delay):
            action = self.agent.select_action(state, 1, train=False)
            next_state, reward, done, info = self.step(action)
            self.replay_buffer.store(state, action, reward, next_state, done)
            state = next_state
            if done:
                state = self.reset_env()

        # Training
        print('Training...')
        state = self.reset_env()
        for frame_idx in range(1, num_frames + 1):
            self._schedule_opponents(frame_idx)
            action = self.agent.select_action(state, frame_idx, train=True)
            next_state, reward, done, info = self.step(action)
            self.replay_buffer.store(state, action, reward, next_state, done)
            state = next_state
            self.tracker.add_frame(reward)
            self.tracker.log('epsilon', self.agent.epsilon_decay(frame_idx))
            if hasattr(self.replay_buffer, 'beta_decay'):
                self.tracker.log('beta', self.replay_buffer.beta_decay(frame_idx))

            # if episode ends
            if done:
                self.tracker.add_game(info)
                state = self.reset_env()

            if frame_idx % self.update_frequency == 0:
                self._update(frame_idx)

            if frame_idx % self.checkpoint_frequency == 0:
                self.checkpoint(frame_idx)

        self.env.close()

    def _add_self(self, frame_idx, p_total=1, rolling=3):
        p = p_total / rolling
        agent = self.agent.copy()
        self.env.add_opponent('self', agent, prob=p, rolling=rolling)
        print(f'!! Added new self-play copy. Frame: {frame_idx} !!')

    def _schedule_opponents(self, frame_idx):
        if self.schedule == 'basic':
            if frame_idx == 500_000:
                self.env.add_basic_opponent(weak=False)
            if frame_idx >= 1_500_000 and frame_idx % 500_000 == 0:
                self._add_self(frame_idx, p_total=3, rolling=3)
        elif self.schedule == 'adv1':
            if frame_idx == 500_000:
                self.env.add_basic_opponent(weak=False)
        elif self.schedule == 'self-play':
            if frame_idx % 500_000 == 0:
                self._add_self(frame_idx, p_total=3, rolling=3)
        elif self.schedule == 'dynamic':
            if frame_idx < 100_000 or len(self.tracker.win_rate_history) < 10:
                return
            
            if frame_idx - self.last_update < 200_000:
                return
            
            win_rate = np.mean(self.tracker.win_rate_history[-10:])
            if win_rate > 70.0:
                self._add_self(frame_idx, p_total=3, rolling=3)
                self.last_update = frame_idx

        else:
            if frame_idx == 500_000:
                self.env.add_basic_opponent(weak=False)

    def _update(self, frame_idx):
        batch = self.replay_buffer.sample_batch_torch(num_frames=frame_idx, device=self.device)
        loss, sample_losses = self.agent.update_model(batch, frame_idx)
        self.tracker.add_update(loss)
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            self.replay_buffer.update_priorities(batch['indices'], sample_losses.cpu().numpy()) 

    def rollout(self, num_games: int):
        self.agent.model.eval()
        state = self.reset_env()
        game_imgs = []
        for _ in range(num_games):
            imgs = [self.env.render(mode='rgb_array')]
            while True:
                action = self.agent.select_action(state)
                state, _, done, _ = self.step(action)
                imgs.append(self.env.render(mode='rgb_array'))
                if done:
                    state = self.reset_env()
                    break
            game_imgs.append(imgs)
        self.agent.model.train()
        return game_imgs
    
    def update_elo(self, frame_idx):
        self.agent.model.eval()
        prev_board = self.tournament.leaderboard.clone()
        self.tournament.evaluate_agent('self', n_games=15)
        elos = self.tournament.leaderboard.elos
        for name, elo in elos.items():
            self.tracker.interval_metrics.add(f'elo/{name}', elo)
        prev_board['self'] = elos['self'] # only update self elo
        self.tournament.leaderboard = prev_board
        self.agent.model.train()

    def checkpoint(self, frame_idx):
        #self.update_elo(frame_idx)
        name = f'frame_{frame_idx:010d}'
        self.agent.save_model(self.model_dir / f'{name}.pt')

        images = self.rollout(4)
        images = [game_imgs + [game_imgs[-1]]*30  for game_imgs in images] # repeat last frame
        images = itertools.chain.from_iterable(images)
        plotting.save_gif(self.model_dir / f'{name}.gif', images)
