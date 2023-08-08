import copy
import itertools
from pathlib import Path
import json

import torch
import torch.nn.functional as F
import numpy as np

from decay import EpsilonDecay
from environments import IcyHockey
from replay_buffer import PrioritizedReplayBuffer, FrameStacker
from tracking import Tracker
from elo_system import HockeyTournamentEvaluation
import plotting

from dqn_stenz import get_stenz

TRAINING_SCHEDULES = ['lilith', 'basic', 'adv1', 'adv2']

class NNAgent:
    ENV = IcyHockey()

    def __init__(self, model, device, frame_stacks=1):
        self.model = model
        self.device = device
        self.stacker = FrameStacker(frame_stacks)

    def reset(self):
        self.stacker.clear()

    def act(self, state):
        state = self.stacker.append_and_stack(state)
        action_discrete = self.select_action(state)
        return self.ENV.discrete_to_continous_action(action_discrete)

    def select_action(self, state, frame_idx=None):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.model(state).argmax(dim=1).item()

    def clone(self):
        model = copy.deepcopy(self.model)
        model.eval().requires_grad_(False)
        return NNAgent(model, self.device, self.stacker.num_frames)

    def save_model(self, path):
        self.model.to('cpu')
        torch.save(self.model, path)
        self.model.to(self.device)

    @classmethod
    def load_model(cls, path, device):
        path = Path(path)
        try:
            with open(path.parent / f'{path.stem}.json', 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            config = None

        if config is None:
            with open(path.parent / 'config.json', 'r') as f:
                config = json.load(f)

        model = torch.load(path, map_location=device)
        model.eval().requires_grad_(False)
        return cls(model, device, frame_stacks=config['frame_stacks'])
    
    @classmethod
    def load_lilith_weak(cls, device):
        path = Path('baselines') / 'lilith_weak.pt'
        return cls.load_model(path, device)

class DqnAgent(NNAgent):

    def __init__(self, model, optimizer, num_actions, device, frame_stacks=1, epsilon_decay=EpsilonDecay(constant_eps=0.1), gamma=0.99, target_update_frequency=1000, no_double=False, scheduler=None):
        super().__init__(model, device, frame_stacks)
        self.target_model = copy.deepcopy(model)
        self.optimizer = optimizer
        self.num_actions = num_actions
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.target_update_frequency = target_update_frequency
        self.num_updates = 0
        self.no_double = no_double
        self.scheduler = scheduler

        self.target_model.requires_grad_(False)
        self.target_model.eval()

    def select_action(self, state, frame_idx=None):
        if frame_idx is not None:
            epsilon = self.epsilon_decay(frame_idx)
        else:
            epsilon = 0.0

        if epsilon > 0.0 and epsilon > np.random.random():
            action = np.random.randint(self.num_actions)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.model(state).argmax(dim=1).item()

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


class DqnTrainer:

    def __init__(self, model_dir, env, agent, replay_buffer, device, frame_stacks=1, training_delay=100_000, update_frequency=1, checkpoint_frequency=100_000, schedule=None):
        assert schedule is None or schedule in ['lilith', 'basic', 'adv1', 'adv2']
        
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

        self.tournament = HockeyTournamentEvaluation()
        self.tournament.add_agent('self', self.agent)
        self.tournament.add_agent('lilith_weak', NNAgent.load_lilith_weak(self.device))
        self.tournament.add_agent('stenz', get_stenz())

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
        self.env.add_opponent('TEMP-AGENT', agent.clone(), prob=4)
        state_self = self.stacker.append_and_stack(state)
        for frame_idx in range(1, num_frames + 1):
            action = agent.select_action(state)
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
            self._schedule_opponents(frame_idx)

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

    def _copy_agent(self):
        model = copy.deepcopy(self.agent.model)
        model.eval().requires_grad_(False)
        return NNAgent(model, self.device, self.agent.stacker.num_frames)

    def _schedule_opponents(self, frame_idx):
        if self.schedule == 'lilith':
            if frame_idx == 500_000:
                self.env.add_basic_opponent(weak=False)
            if frame_idx >= 1_500_000 and frame_idx % 100_000 == 0:
                agent = self._copy_agent()
                self.env.add_opponent('self', agent, prob=5, rolling=5)
        elif self.schedule == 'basic':
            if frame_idx == 500_000:
                self.env.add_basic_opponent(weak=False)
            if frame_idx == 1:
                agent = NNAgent.load_lilith_weak(self.device)
                self.env.add_opponent('lilith_weak', agent, prob=5)
            if frame_idx >= 2_000_000 and frame_idx % 200_000 == 0:
                agent = self._copy_agent()
                self.env.add_opponent('self', agent, prob=5, rolling=8)
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
        self.update_elo(frame_idx)

        name = f'frame_{frame_idx:010d}'
        self.agent.save_model(self.model_dir / f'{name}.pt')

        images = self.rollout(4)
        images = [game_imgs + [game_imgs[-1]]*30  for game_imgs in images] # repeat last frame
        images = itertools.chain.from_iterable(images)
        plotting.save_gif(self.model_dir / f'{name}.gif', images)
