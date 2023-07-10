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
    return np.sign(angvel) * np.log(1 + 0.5*np.fabs(angvel))

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
            if i == len(layers) - 1:
                self.layers.append(nn.Linear(layers[i-1], layers[i]))
            else:
                self.layers.append(MLPLayer(layers[i-1], layers[i], activation, norm_layer)) 

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EMAModel(nn.Module):

    def __init__(self, model, base_model, momentum=0.9):
        super().__init__()
        #self.model = copy.deepcopy(base_model).requires_grad_(True)
        self.model = model
        self.momentum = momentum

        for param, base_param in zip(self.model.parameters(), base_model.parameters()):
            param.data = base_param.data

        self.model.requires_grad_(True)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
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
        self.n = 0

    def act(self, obs):
        obs = normalize_(obs)
        obs = torch.from_numpy(obs).float().to(self.device)
        Q = self.module(obs)
        action = torch.argmax(Q)
        if self.module.training and np.random.rand() < max(0.1, 0.8 - 0.8*self.n / 1e6):
            action = torch.randint(0, 7, ())
        action = action.detach().cpu().numpy()
        action_cont = [(action == 1) * -1 + (action == 2) * 1,  # player x
                    (action == 3) * -1 + (action == 4) * 1,  # player y
                    (action == 5) * -1 + (action == 6) * 1]  # player angle
        self.last_action = action_cont
        self.last_action_discrete = action
        self.n += 1
        return self.last_action

    def reset(self):
        self.last_action = None

@torch.no_grad()
def play_game(env, agent1, agent2, max_steps = 1500, render=True, action_repeats=1):
    obs_agent1, info = env.reset()
    obs_agent2 = env.obs_agent_two()
    states = []
    for _ in range(max_steps):
        if render:
            states.append(Image.fromarray(env.render('rgb_array')))
        a1 = agent1.act(obs_agent1)
        a2 = agent2.act(obs_agent2)
        r_cum = 0
        for _ in range(action_repeats):
            obs_agent1, r, d, _, info = env.step(np.hstack([a1,a2]))
            obs_agent2 = env.obs_agent_two()
            r_cum += r
            if d:
                break
        if d: 
            break
    return r, states

def DQN():
    BS = 64
    GAMMA = 0.995
    LR = 1e-4
    ACTION_REPEATS = 3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLP([18, 64, 64, 7]).to(device)
    ema_model = EMAModel(MLP([18, 64, 64, 7]).to(device), model, momentum=0.9999) #
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(200000)

    env = lh.LaserHockeyEnv()
    agent1 = NNAgent(model, device, action_repeats=ACTION_REPEATS)
    agent2 = lh.BasicOpponent()
    agent_bootstrap = lh.BasicOpponent()

    # Prepopulate replay buffer
    for game in range(100):
        if game < 25:
            mode = lh.LaserHockeyEnv.TRAIN_SHOOTING
        elif game < 50:
            mode = lh.LaserHockeyEnv.TRAIN_DEFENSE
        else:
            mode = lh.LaserHockeyEnv.NORMAL
        
        obs_agent1, info = env.reset(mode=mode)
        obs_agent2 = env.obs_agent_two()
        for i in range(1000):
            a1 = agent_bootstrap.act(obs_agent1)
            a2 = agent2.act(obs_agent2)
            r_cum = 0.0
            for _ in range(ACTION_REPEATS):
                obs_agent1_new, r, d, _, info = env.step(np.hstack([a1,a2]))
                obs_agent2_new = env.obs_agent_two()
                r_cum += r
                if d: 
                    break

            x_forward = a1[0] > 0.5
            x_backward = a1[0] < -0.5
            y_forward = a1[1] > 0.5
            y_backward = a1[1] < -0.5
            angle_forward = a1[2] > 0.5
            angle_backward = a1[2] < -0.5 
            action = 0
            if angle_backward: action = 5
            if angle_forward: action = 6
            if x_backward: action = 1
            if x_forward: action = 2
            if y_backward: action = 3
            if y_forward: action = 4

            r_cum -= 0.2
            sample = (obs_agent1, action, r_cum, obs_agent1_new, d)
            replay_buffer.add(sample)

            obs_agent1 = obs_agent1_new
            obs_agent2 = obs_agent2_new

            if d:
                break

    print(len(replay_buffer.buffer))
    
    model.train()
    ema_model.train()
    for game in range(30000):
        if game < 600 and game % 2 == 0:
            mode = lh.LaserHockeyEnv.TRAIN_SHOOTING
        elif game < 600:
            mode = lh.LaserHockeyEnv.TRAIN_DEFENSE
        else:
            mode = lh.LaserHockeyEnv.NORMAL
        print(f'Game {game}')
        obs_agent1, info = env.reset(mode=mode)
        obs_agent2 = env.obs_agent_two()
        agent1.reset()
        Qs = []
        for i in range(1000):
            with torch.no_grad():
                a1 = agent1.act(obs_agent1)
            a2 = agent2.act(obs_agent2)

            r_cum = 0.0
            for _ in range(ACTION_REPEATS):
                obs_agent1_new, r, d, _, info = env.step(np.hstack([a1,a2]))
                obs_agent2_new = env.obs_agent_two()
                r_cum += r
                if d: 
                    break

            r_cum -= 0.01
            if r_cum != 0 or random.random() < 0.05:
                sample = (obs_agent1, agent1.last_action_discrete, r_cum, obs_agent1_new, d)
                replay_buffer.add(sample)

            obs_agent1 = obs_agent1_new
            obs_agent2 = obs_agent2_new

            if len(replay_buffer.buffer) > 500 and i % 1 == 0:
                optimizer.zero_grad()

                batch = replay_buffer.sample(BS)
                obs, action, r, obs_next, d_ = zip(*batch)
                obs = normalize(np.array(obs))
                obs_next = normalize(np.array(obs_next))
                obs = torch.from_numpy(obs).float().to(device)
                action = torch.from_numpy(np.array(action)).long().to(device)
                r = torch.from_numpy(np.array(r)).float().to(device)
                obs_next = torch.from_numpy(obs_next).float().to(device)
                d_ = torch.from_numpy(np.array(d_)).float().to(device)

                Q = model(obs)
                #print('A', Q.mean())
                Q_next = ema_model(obs_next)
                #print('B', Q_next.mean())
                Q_next_max = torch.max(Q_next, dim=1)[0]
                Q_target = r + GAMMA * Q_next_max * (1 - d_)
                Qs += [Q_target.detach().mean().cpu().item()]

                loss = torch.mean((Q_target - Q.gather(1, action.unsqueeze(1))) ** 2)
                loss.backward()
                optimizer.step()
                ema_model.update(model)

            if d:
                break
    
        if Qs:
            print(f'Q: {np.mean(Qs)}')
        if game % 500 == 0:
            model.train()
            agent1.reset()
            r, states = play_game(env, agent1, agent2,action_repeats=ACTION_REPEATS)
            states[0].save(f'game{game}.gif', save_all=True, append_images=states[1:], duration=(1/50)*1000*ACTION_REPEATS, loop=0)
            model.train()


def main():
    env = lh.LaserHockeyEnv(mode=lh.LaserHockeyEnv.TRAIN_SHOOTING)
    player1 = lh.BasicOpponent()
    player2 = lh.BasicOpponent()
    r, states = play_game(env, player1, player2)
    states[0].save('basic_game.gif', save_all=True, append_images=states[1:], duration=(1/50)*1000, loop=0)

if __name__ == '__main__':
    #main()
    DQN()