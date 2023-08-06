import argparse
import wandb
import torch
import torch.nn.functional as F
import numpy as np

from tracking import Tracker
from replay_buffer import FrameStacker2


class DqnAgent:

    def __init__(self, model, optimizer, num_actions, target_update_frequency=1000):
        self.model = model
        self.optimizer = optimizer
        self.num_actions = num_actions
        self.target_update_frequency = target_update_frequency

    def select_action(self, state):
        """Select an action from the input state and return state and action."""
        if self.epsilon > np.random.random():
            selected_action = np.random.randint(self.num_actions)
        else:
            selected_action = self.dqn(
                torch.FloatTensor(state).unsqueeze(0).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        return selected_action

    def update_model(self, samples):
        samples = batch
        # PER needs beta to calculate weights
        weights = samples["weights"].unsqueeze(1)
        indices = samples["indices"] # nd.array

        # N-step Learning loss
        gamma = self.gamma ** self.n_step
        elementwise_loss = self.gamma(samples, gamma)

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        #clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        return loss.item()

    def compute_loss(self, samples, gamma):
        """Return the loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"]).to(device)
        done = torch.FloatTensor(samples["done"]).to(device)

        action = action.unsqueeze(1)  # B x 1
        curr_q_value = self.dqn(state).gather(1, action)[:,0] # B
        policy_actions = self.dqn(next_state).argmax(dim=1, keepdim=True) # B x 1
        next_q_value = self.dqn_target(next_state).gather(1, policy_actions)[:,0].detach() # B
        mask = 1 - done # B
        target = (reward + self.gamma * next_q_value * mask).to(self.device)
        loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")
        return loss

class DqnTrainer:

    def __init__(self, env, agent, replay_buffer, optimizer, device, epsilon_decay, training_delay=1_000_000):
        self.env = env
        self.agent = agent
        self.optimizer = optimizer
        self.device = device

        self.replay_buffer = replay_buffer
        self.stacker = FrameStacker2(1)
        
        self.tracker = Tracker()
        self.score = 0

        self.epsilon_decay = epsilon_decay
        self.training_delay = training_delay

    def reset_env(self):
        self.score = 0
        self.stacker.clear()
        state = self.env.reset()
        return self.stacker.append_and_stack(state)

    def step(self, action):
        next_state, reward, done, _, info = self.env.step(action)
        next_state = self.stacked.append_and_stack(next_state)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)  # store a full transition

        return next_state, reward, done, info


    def train(self, num_frames: int):
        """Train the agent."""
        self.is_test = False

        state = self.reset_env()

        update_cnt = 0  # counts the number of steps between each update
        losses = []  # loss for each training step
        scores = []  # score for each episode
        frame_scores = []  # average score each frame_interval frames
        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done, info = self.step(action)
            self.replay_buffer.store(state, action, reward, next_state, done)
            state = next_state
            self.score += reward

            # if episode ends
            if done:
                self.tracker.finish_game(self.score, info['winner'])
                state = self.reset_env()

            if frame_idx > self.training_delay and frame_idx % 2 == 0:
                batch = self.replay_buffer.sample_batch_torch(num_frames=frame_idx, device=self.device)
                loss = self.agent.update_model(batch)

                if update_cnt % self.target_update == 0 and not self.no_double:
                    self._target_hard_update()

            if frame_idx % self.frame_interval == 0:
                # print
    
        self.env.close()

        return frame_scores, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-wandb', action='store_true', default=True)
    args = parser.parse_args()

    wandb_mode = 'disabled' if args.no_wandb else 'online'
    wandb.init(project='ice', mode=wandb_mode)


if __name__ == '__main__':
    main()