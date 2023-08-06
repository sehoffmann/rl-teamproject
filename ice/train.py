import argparse
import wandb
import torch
import torch.nn.functional as F
import numpy as np
import copy

from tracking import Tracker
from replay_buffer import FrameStacker2, ReplayBuffer, PrioritizedReplayBuffer
from discretization import DiscreteHockey_BasicOpponent
from models import DenseNet
from decay import EpsilonDecay

class DqnAgent:

    def __init__(self, model, optimizer, num_actions, device, epsilon_decay=EpsilonDecay(constant_eps=0.1), gamma=0.99, target_update_frequency=1000):
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.optimizer = optimizer
        self.num_actions = num_actions
        self.device = device
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.target_update_frequency = target_update_frequency
        self.num_updates = 0

    def select_action(self, state, frame_idx=None):
        """Select an action from the input state and return state and action."""
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

        policy_actions = self.model(next_state).argmax(dim=1, keepdim=True) # B x 1
        next_q_value = self.target_model(next_state).gather(1, policy_actions) # B x 1
        next_q_value = next_q_value.squeeze(1).detach() # B
        
        mask = 1 - done # B
        target = (reward + self.gamma * next_q_value * mask).to(self.device)
        loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")
        return loss

class DqnTrainer:

    def __init__(self, env, agent, replay_buffer, device, frame_stacks=1, training_delay=100_000, update_frequency=1, track_frequency=10_000):
        self.env = env
        self.agent = agent
        self.device = device
        self.replay_buffer = replay_buffer
        self.training_delay = training_delay
        self.track_frequency = track_frequency
        self.update_frequency = update_frequency

        self.stacker = FrameStacker2(frame_stacks)        
        self.tracker = Tracker()
        self._score = 0


    def reset_env(self):
        self._score = 0
        self.stacker.clear()
        state = self.env.reset()
        return self.stacker.append_and_stack(state)

    def step(self, action):
        next_state, reward, done, _, info = self.env.step(action)
        next_state = self.stacker.append_and_stack(next_state)
        return next_state, reward, done, info

    def train(self, num_frames: int):
        state = self.reset_env()

        for frame_idx in range(1, num_frames + 1):
            action = self.agent.select_action(state, frame_idx)
            next_state, reward, done, info = self.step(action)
            self.replay_buffer.store(state, action, reward, next_state, done)
            state = next_state
            self._score += reward

            # if episode ends
            if done:
                self.tracker.finish_game(self._score, info['winner'])
                state = self.reset_env()

            if frame_idx > self.training_delay and frame_idx % self.update_frequency == 0:
                batch = self.replay_buffer.sample_batch_torch(num_frames=frame_idx, device=self.device)
                loss, sample_losses = self.agent.update_model(batch, frame_idx)
                self.replay_buffer.update_priorities(batch['indices'], sample_losses.cpu().numpy()) 

            if frame_idx % self.track_frequency == 0:
                print(frame_idx, self.tracker.num_games)
                # print
    
        self.env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-wandb', action='store_true', default=True)
    args = parser.parse_args()

    wandb_mode = 'disabled' if args.no_wandb else 'online'
    wandb.init(project='ice', mode=wandb_mode)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    warmup_frames = 30_000

    frame_stacks = 1
    buffer_size = 500_000
    batch_size = 128
    lr = 1e-4
    n_step = 3
    gamma = 0.99
    eps_decay_frames = 1_000_000
    beta_decay_frames = 3_000_000
    update_frequency = 2

    # ENV
    env = DiscreteHockey_BasicOpponent()
    
    # Replay Buffer
    obs_shape = [env.observation_space.shape[0] * frame_stacks]
    replay_buffer = PrioritizedReplayBuffer(
        obs_shape, 
        buffer_size, 
        batch_size, 
        n_step = n_step,
        gamma = gamma,
        beta_frames = beta_decay_frames
    )

    # Model & DQN Agent
    num_actions = env.action_space.n
    model = DenseNet(
        obs_shape[0], 
        num_actions, 
        hidden_size=256, 
        no_dueling=False
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epsilon_decay = EpsilonDecay(num_frames=eps_decay_frames)
    dqn_agent = DqnAgent(
        model, 
        optimizer, 
        num_actions,
        device,
        epsilon_decay=epsilon_decay, 
        gamma=gamma**n_step,
    )

    # Trainer
    trainer = DqnTrainer(
        env, 
        dqn_agent, 
        replay_buffer, 
        device,
        frame_stacks=frame_stacks,
        update_frequency=update_frequency,
        training_delay=warmup_frames,
    )

    trainer.train(2_000_000)


if __name__ == '__main__':
    main()