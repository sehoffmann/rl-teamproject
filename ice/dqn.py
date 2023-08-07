import copy
import itertools
import torch
import torch.nn.functional as F
import numpy as np

from decay import EpsilonDecay
from environments import DiscreteHockey_BasicOpponent
from replay_buffer import PrioritizedReplayBuffer, FrameStacker
from rollout_collection import RemoteReplayBuffer
from tracking import Tracker
from elo_system import HockeyTournamentEvaluation
import plotting

from dqn_stenz import get_stenz

def discrete_to_cont_action(discrete_action):
    # TODO: better place for this??
    return DiscreteHockey_BasicOpponent().discrete_to_continous_action(discrete_action)

class DqnAgent:

    def __init__(self, model, optimizer, num_actions, device, frame_stacks=1, epsilon_decay=EpsilonDecay(constant_eps=0.1), gamma=0.99, target_update_frequency=1000, no_double=False, scheduler=None):
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.optimizer = optimizer
        self.num_actions = num_actions
        self.device = device
        self.stacker = FrameStacker(frame_stacks)  # required for rollout
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.target_update_frequency = target_update_frequency
        self.num_updates = 0
        self.no_double = no_double
        self.scheduler = scheduler

        self.target_model.requires_grad_(False)
        self.target_model.eval()

    def reset(self):
        self.stacker.clear()

    def act(self, state):
        state = self.stacker.append_and_stack(state)
        return discrete_to_cont_action(self.select_action(state))

    def select_action(self, state, frame_idx=None):
        if frame_idx is not None:
            epsilon = self.epsilon_decay(frame_idx)
        else:
            epsilon = 0.0

        if epsilon > 0.0 and epsilon > np.random.random():
            action = np.random.randint(self.num_actions)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.model(state).argmax(dim=1)
            action = action.item()

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

    def compute_loss(self, samples):
        state = samples['obs']
        next_state = samples['next_obs']
        action = samples['acts']
        reward = samples['rews']
        done = samples['done']

        action = action.unsqueeze(1)  # B x 1
        curr_q_value = self.model(state).gather(1, action) # B x 1
        curr_q_value = curr_q_value.squeeze(1) # B

        next_q_value = self.target_model(next_state)

        if self.no_double:
            best_action = next_q_value.argmax(dim=1, keepdim=True) # B x 1
        else:
            best_action = self.model(next_state).argmax(dim=1, keepdim=True) # B x 1
        
        next_value_f = next_q_value.gather(1, best_action) # B x 1
        next_value_f = next_value_f.squeeze(1).detach() # B
        
        mask = 1 - done # B
        target = (reward + self.gamma * next_value_f * mask)
        loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")
        return loss

    def save_model(self, path):
        self.model.to('cpu')
        torch.save(self.model, path)
        self.model.to(self.device)


class DqnTrainer:

    def __init__(self, model_dir, env, agent, replay_buffer, device, frame_stacks=1, training_delay=100_000, update_frequency=1, checkpoint_frequency=100_000, agent_name="buster_blader", populate_replay_online=False):
        self.model_dir = model_dir
        self.env = env
        self.agent = agent
        self.device = device
        self.replay_buffer = replay_buffer
        self.training_delay = training_delay
        self.update_frequency = update_frequency
        self.checkpoint_frequency = checkpoint_frequency

        self.stacker = FrameStacker(frame_stacks)        
        self.tracker = Tracker()

        self.tournament = HockeyTournamentEvaluation(restart=True)
        self.tournament.register_agent(agent_name, self.agent)
        self.agent_name = agent_name
        #self.tournament.register_agent("stenz", get_stenz(), n_welcome_games=10)
        self.populate_replaty_online = populate_replay_online
        self.remote_replay_buffer = RemoteReplayBuffer(obs_dim=[self.env.observation_space.shape[0] * frame_stacks], size=100_000, batch_size=32, n_step=1, gamma=0.99)


    def reset_env(self):
        self.stacker.clear()
        state = self.env.reset()
        return self.stacker.append_and_stack(state)

    def step(self, action):
        next_state, reward, done, _, info = self.env.step(action)
        next_state = self.stacker.append_and_stack(next_state)
        return next_state, reward, done, info

    def train(self, num_frames: int):
        

        # Warmup
        print('Filling replay buffer...')
        state = self.reset_env()   
        for _ in range(self.training_delay):
            action = self.agent.select_action(state, 1)
            next_state, reward, done, info = self.step(action)
            self.replay_buffer.store(state, action, reward, next_state, done)
            state = next_state
            if done:
                state = self.reset_env()

        # Training
        print('Training...')
        state = self.reset_env()
        for frame_idx in range(1, num_frames + 1):
            action = self.agent.select_action(state, frame_idx)
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

    def _update(self, frame_idx):
        if np.random.random() < 0.5 and self.populate_replaty_online:
            batch = self.remote_replay_buffer.sample_batch_torch(num_frames=frame_idx, device=self.device)
        else:
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
        self.tournament.evaluate_agent(self.agent_name, self.agent, n_games=10)
        self.tracker.add_checkpoint(self.tournament.elo_leaderboard.elo_system)
        self.agent.model.train()

    def checkpoint(self, frame_idx):
        self.update_elo(frame_idx)

        name = f'frame_{frame_idx:010d}'
        self.agent.save_model(self.model_dir / f'{name}.pt')

        images = self.rollout(4)
        images = [game_imgs + [game_imgs[-1]]*30  for game_imgs in images] # repeat last frame
        images = itertools.chain.from_iterable(images)
        plotting.save_gif(self.model_dir / f'{name}.gif', images)
