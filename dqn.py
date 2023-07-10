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
import pandas as pd
import matplotlib.pyplot as plt

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
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.act = activation()
        self.norm = nn.Identity()#norm_layer(out_channels)

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

        self.model.requires_grad_(False)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def update(self, base_model):
        for param, base_param in zip(self.model.parameters(), base_model.parameters()):
            param.data = self.momentum * base_param.data + (1 - self.momentum) * param.data


class NNAgent:

    def __init__(self, module, device, action_repeats = 1, epsilon=0.8, epsilon_decay=1e-6, min_epsilon=0.1, max_sampling=False):
        self.module = module
        self.device = device
        self.action_repeats = action_repeats
        self.last_action = None
        self.last_action_discrete = None
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.max_sampling = max_sampling
        self.n = 0

    def _choose_action(self, Q):
        if self.max_sampling:
            action = torch.argmax(Q)
        else:
            p = torch.softmax(Q, dim=-1)
            action = torch.multinomial(p, 1).squeeze(-1)

        p = np.random.rand()
        threshold = self.epsilon - self.epsilon_decay*self.n
        threshold = max(threshold, self.min_epsilon)
        if self.module.training and p < threshold:
            action = torch.randint(0, 7, ())
        return action.detach().cpu().numpy()

    def _action_to_cont(self, action):
        action_cont = [(action == 1) * -1 + (action == 2) * 1,  # player x
                    (action == 3) * -1 + (action == 4) * 1,  # player y
                    (action == 5) * -1 + (action == 6) * 1]  # player angle
        return action_cont

    def act(self, obs):
        obs = normalize_(obs)
        obs = torch.from_numpy(obs).float().to(self.device)
        Q = self.module(obs)
        action = self._choose_action(Q)
        self.last_action_discrete = action
        self.last_action = self._action_to_cont(action)
        self.n += 1
        return self.last_action

    def reset(self):
        self.last_action = None


class Tracker:
    def __init__(self):
        self.values = {}

    def add(self, key, value, step):
        if key not in self.values:
            self.values[key] = []
        self.values[key].append((step, value))

    def save_csv(self, path):
        df = pd.DataFrame()
        for key in self.values:
            df[key] = pd.Series([v[1] for v in self.values[key]], [v[0] for v in self.values[key]])
        df.to_csv(path)

    def plot(self, keys, title=None, smoothing=0.0):
        fig = plt.figure(figsize=(12,6))
        for key in keys:
            x = [v[0] for v in self.values[key]]
            vals = [v[1] for v in self.values[key]]
            if smoothing > 0:
                vals = pd.Series(vals).ewm(alpha=1-smoothing).mean()
            plt.plot(x, vals, label=key)
        plt.legend()
        plt.title(title)
        return fig

def prepopulate(replay_buffer, action_repeats):
    env = lh.LaserHockeyEnv()
    agent1 = lh.BasicOpponent()
    agent2 = lh.BasicOpponent()

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
            a1 = agent1.act(obs_agent1)
            a2 = agent2.act(obs_agent2)
            r_cum = 0.0
            for _ in range(action_repeats):
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

            sample = (obs_agent1, action, r_cum, obs_agent1_new, d)
            replay_buffer.add(sample)

            obs_agent1 = obs_agent1_new
            obs_agent2 = obs_agent2_new

            if d:
                break

@torch.no_grad()
def play_game(agent1, agent2, max_steps = 1500, render=True, action_repeats=1):
    env = lh.LaserHockeyEnv()
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

def evaluate(agent, game, tracker, action_repeats, opponent=None, N=20):
    if opponent is None:
        opponent = lh.BasicOpponent()

    r, states = play_game(agent, opponent, action_repeats=action_repeats)
    states[0].save(f'game{game}.gif', save_all=True, append_images=states[1:], duration=(1/50)*1000*action_repeats, loop=0)

    wins, draws, losses = (0,0,0)
    for _ in range(20):
        r, _ = play_game(agent, opponent, action_repeats=action_repeats, render=False)
        if r > 0:
            wins += 1
        elif r == 0:
            draws += 1
        else:
            losses += 1
    tracker.add('win_rate', wins/N, game)
    tracker.add('draw_rate', draws/N, game)
    tracker.add('loss_rate', losses/N, game)

    tracker.save_csv('results.csv')
    tracker.plot(['win_rate', 'draw_rate', 'loss_rate'], title='Skill')
    plt.savefig('skill.png')
    plt.close()
    tracker.plot(['value_f'], title='Avg. Value Function', smoothing=0.00)
    plt.savefig('value.png')
    plt.close()
    tracker.plot(['td_error'], title='TD Error / Loss', smoothing=0.00)
    plt.savefig('loss.png')
    plt.close()

def DQN():
    BS = 64
    GAMMA = 0.995
    LR = 1e-4
    ACTION_REPEATS = 3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tracker = Tracker()
    model = MLP([18, 64, 64, 7]).to(device)
    ema_model = EMAModel(MLP([18, 64, 64, 7]).to(device), model, momentum=0.995)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(200000)

    env = lh.LaserHockeyEnv()
    agent = NNAgent(model, device, action_repeats=ACTION_REPEATS)
    opponent = lh.BasicOpponent()

    prepopulate(replay_buffer, ACTION_REPEATS)
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

        obs_agent1, info = env.reset(mode=mode)
        obs_agent2 = env.obs_agent_two()
        agent.reset()
        
        n_updates = 0
        Qs = []
        td_errors = []
        for i in range(1000//ACTION_REPEATS):
            with torch.no_grad():
                a1 = agent.act(obs_agent1)
            a2 = opponent.act(obs_agent2)

            r_cum = -(ACTION_REPEATS/1000)
            for _ in range(ACTION_REPEATS):
                obs_agent1_new, r, d, _, info = env.step(np.hstack([a1,a2]))
                obs_agent2_new = env.obs_agent_two()
                r_cum += r
                if d: 
                    break

            if r_cum != 0 or random.random() < 0.5:
                sample = (obs_agent1, agent.last_action_discrete, r_cum, obs_agent1_new, d)
                replay_buffer.add(sample)

            obs_agent1 = obs_agent1_new
            obs_agent2 = obs_agent2_new

            if  i % 1 == 0:
                n_updates += 1
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
                Q_next = ema_model(obs_next)
                Q_next_max = torch.max(Q_next, dim=1)[0]
                Q_target = r + GAMMA * Q_next_max * (1 - d_)

                loss = torch.mean((Q_target - Q.gather(1, action.unsqueeze(1))) ** 2)
                loss.backward()
                optimizer.step()
                ema_model.update(model)

                Qs += [Q_target.detach().mean().cpu().item()]
                td_errors += [loss.detach().cpu().item()]

            if d:
                break
    
        print(f'Game {game}: {n_updates} updates')
        if n_updates > 0:
            tracker.add('value_f', np.mean(Qs), game)
            tracker.add('td_error', np.mean(td_errors), game)
    
        if game % 200 == 0:
            print('N Steps:', agent.n)
            evaluate(agent, game, tracker, ACTION_REPEATS, opponent=opponent)


def main():
    env = lh.LaserHockeyEnv(mode=lh.LaserHockeyEnv.TRAIN_SHOOTING)
    player1 = lh.BasicOpponent()
    player2 = lh.BasicOpponent()
    r, states = play_game(env, player1, player2)
    states[0].save('basic_game.gif', save_all=True, append_images=states[1:], duration=(1/50)*1000, loop=0)

if __name__ == '__main__':
    #main()
    DQN()