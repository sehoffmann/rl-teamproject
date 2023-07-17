import copy
import torch
import random
from torch import nn
import numpy as np
import time
import laserhockey.laser_hockey_env as lh
import pandas as pd
from laserhockey.hockey_env import HumanOpponent
from PIL import Image
from gymnasium.spaces import Box, Discrete

"""
The Environment:

Rewards: Either -10 or +10 for losing or winning the game, respectively. 0 otherwise.
Each player has a position, velocity, angle, and angular velocity.
There a alot of frames, so it might make sense to only look at every 5th frame or so.

W,H = 20, 13.33
Center = (10, 6.666)

Max puck speed: 20
Max player speed: 10

State-Space: 18-dimensional vector
    * Self-Pos relative to center
        - x: [-10, 10]
        - y: [-6.666, 6.666]
    * Self-Angle as sin and cos
        - sin: [-1, 1]
        - cos: [-1, 1]
    * Self-Vel 2D
        - [-10, 10]
    * Self-AngularVel
        - Unlimited, but realistic values are [-5, 5]
    * Opponent-Pos relative to center
    * Opponent-Angle as sin and cos
    * Opponent-Vel 2D
    * Opponent-AngularVel 1D
    * Puck-Pos relative to center 
        - x: [-10, 10]
        - y: [-6.666, 6.666]
    * Puck-Vel 2D

    - Players are swapped for observation of player 2.

Action-Space: 3-dimensional vector
    * Bounded to [-1,1]
    * Target-Pos X
    * Target-Pos Y
    * Target-Angle

    - These are all forces / torques applied to the player.
    - I.e. accelarations.
"""

POS_BOUNDS = np.array([10, 6.666])

def normalize_pos(pos):
    return pos / POS_BOUNDS

def normalize_vel(vel):
    return vel / 10

def normalize_angvel(angvel):
    return angvel / 5

def normalize_(obs):
    # Player 1
    obs[..., 0:2] = normalize_pos(obs[..., 0:2])
    obs[..., 4:6] = normalize_vel(obs[..., 4:6])
    obs[..., 6] = normalize_angvel(obs[..., 6])

    # Player 2
    obs[..., 7:9] = normalize_pos(obs[..., 7:9])
    obs[..., 11:13] = normalize_vel(obs[..., 11:13])
    obs[..., 13] = normalize_angvel(obs[..., 13])

    # Puck
    obs[..., 14:16] = normalize_pos(obs[..., 14:16])
    obs[..., 16:18] = normalize_vel(obs[..., 16:18])
    return obs

def normalize(obs):
    obs = obs.copy()
    return normalize_(obs)


class ReplayBuffer:

    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.position = 0

    def add(self, sample):
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.position] = sample
        self.position = (self.position + 1) % self.size

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        return [self.buffer[i] for i in indices]

class MLPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.act = activation()
        self.norm = norm_layer(out_channels)

    def forward(self, x):
        return self.act(self.norm(self.linear(x)))

class MLP(nn.Module):

    def __init__(self, layers, activation=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        assert len(layers) >= 2
        
        self.layers = nn.ModuleList()
        for i in range(1, len(layers)):
            self.layers.append(MLPLayer(layers[i-1], layers[i], activation, norm_layer)) 

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EMAModel(nn.Module):

    def __init__(self, base_model, momentum=0.9):
        super().__init__()
        self.model = copy.deepcopy(base_model)
        self.momentum = momentum

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def update(self, base_model):
        for param, base_param in zip(self.model.parameters(), base_model.parameters()):
            param.data = self.momentum * base_param.data + (1 - self.momentum) * param.data


class NNAgent:

    def __init__(self, module, device, action_repeats = 1):
        self.module = module
        self.device = device
        self.action_repeats = action_repeats
        self.last_action = None
        self.last_action_discrete = None
        self.i = 0

    def act(self, obs):
        if self.i % self.action_repeats == 0:
            obs = normalize_(obs)
            obs = torch.from_numpy(obs).float().to(self.device)
            Q = self.module(obs)
            action = torch.argmax(Q)
            if self.module.training and np.random.rand() < 0.1:
                action = torch.randint(0, 7, ())
            action = action.detach().numpy()
            action_cont = [(action == 1) * -1 + (action == 2) * 1,  # player x
                       (action == 3) * -1 + (action == 4) * 1,  # player y
                       (action == 5) * -1 + (action == 6) * 1]  # player angle
            self.last_action = action_cont
            self.last_action_discrete = action
        self.i += 1
        return self.last_action

    def reset(self):
        self.last_action = None
        self.i = 0

def play_game(env, agent1, agent2, max_steps = 1500, render=True):
    obs_agent1, info = env.reset()
    obs_agent2 = env.obs_agent_two()
    states = []
    for _ in range(max_steps):
        if render:
            states.append(Image.fromarray(env.render('rgb_array')))
        a1 = agent1.act(obs_agent1)
        a2 = agent2.act(obs_agent2)
        obs_agent1, r, d, _, info = env.step(np.hstack([a1,a2]))
        obs_agent2 = env.obs_agent_two()
        if d: break
    return r, states

def DQN():
    BS = 32
    GAMMA = 0.9
    LR = 1e-3
    ACTION_REPEATS = 5

    model = MLP([18, 64, 7])
    ema_model = EMAModel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(10000)

    env = lh.LaserHockeyEnv()
    agent1 = NNAgent(model, 'cpu', action_repeats=ACTION_REPEATS)
    agent2 = lh.BasicOpponent()
    
    model.train()
    ema_model.train()
    for game in range(60):
        if game < 20:
            mode = lh.LaserHockeyEnv.TRAIN_SHOOTING
        elif game < 40:
            mode = lh.LaserHockeyEnv.TRAIN_DEFENSE
        else:
            mode = lh.LaserHockeyEnv.NORMAL
        print(f'Game {game}')
        obs_agent1, info = env.reset(mode=mode)
        obs_agent2 = env.obs_agent_two()
        agent1.reset()
        for i in range(500):
            a1 = agent1.act(obs_agent1)
            a2 = agent2.act(obs_agent2)
            obs_agent1_new, r, d, _, info = env.step(np.hstack([a1,a2]))
            obs_agent2_new = env.obs_agent_two()

            if r != 0 or random.random() < 0.1:
                sample = (obs_agent1, agent1.last_action_discrete, r, obs_agent1_new, d)
                replay_buffer.add(sample)

            obs_agent1 = obs_agent1_new
            obs_agent2 = obs_agent2_new

            if d: break

            if len(replay_buffer.buffer) > 500 and i % (3*ACTION_REPEATS) == 0:
                optimizer.zero_grad()

                batch = replay_buffer.sample(BS)
                obs, action, r, obs_next, d = zip(*batch)
                obs = normalize(np.array(obs))
                obs_next = normalize(np.array(obs_next))
                obs = torch.from_numpy(obs).float()
                action = torch.from_numpy(np.array(action)).long()
                r = torch.from_numpy(np.array(r)).float()
                obs_next = torch.from_numpy(obs_next).float()
                d = torch.from_numpy(np.array(d)).float()

                Q = model(obs)
                Q_next = ema_model(obs_next)
                Q_next_max = torch.max(Q_next, dim=1)[0]
                Q_target = r + GAMMA * Q_next_max * (1 - d)
                print(Q_target.mean())

                loss = torch.mean((Q_target - Q.gather(1, action.unsqueeze(1))) ** 2)
                loss.backward()
                optimizer.step()
                ema_model.update(model)

    model.eval()
    r, states = play_game(env, agent1, agent2)
    states[0].save('basic_game.gif', save_all=True, append_images=states[1:], duration=(1/50)*1000, loop=0)



def main():
    env = lh.LaserHockeyEnv(mode=lh.LaserHockeyEnv.TRAIN_SHOOTING)
    player1 = lh.BasicOpponent()
    player2 = lh.BasicOpponent()
    r, states = play_game(env, player1, player2)
    states[0].save('basic_game.gif', save_all=True, append_images=states[1:], duration=(1/50)*1000, loop=0)


if __name__ == '__main__':
    #main()
    DQN()
    # baselines()