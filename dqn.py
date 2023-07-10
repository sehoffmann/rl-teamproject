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

POS_BOUNDS = torch.tensor([10, 6.666])

def normalize_pos(pos):
    return pos / torch.tensor([10, 6.666], dtype=torch.float, device=pos.device)

def normalize_vel(vel):
    return vel / 10

def normalize_angvel(angvel):
    return torch.sign(angvel) * torch.log(1 + 0.5*torch.abs(angvel))

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
        self.states_0 = []
        self.states_1 = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.position = 0

    def __len__(self):
        return len(self.states_0)
    
    def __getitem__(self, index):
        return self.states_0[index], self.states_1[index], self.actions[index], self.rewards[index], self.dones[index]
    
    def __setitem__(self, index, value):
        self.states_0[index], self.states_1[index], self.actions[index], self.rewards[index], self.dones[index] = value

    def _extend(self, size=1):
        self.states_0.extend([None] * size)
        self.states_1.extend([None] * size)
        self.actions.extend([None] * size)
        self.rewards.extend([None] * size)
        self.dones.extend([None] * size)

    def add(self, state_0, state_1, action, reward, done):
        if len(self) < self.size:
            self._extend(1)
        state_0  = torch.as_tensor(state_0, dtype=torch.float)
        state_1  = torch.as_tensor(state_1, dtype=torch.float)
        action   = torch.as_tensor(action, dtype=torch.float)
        reward   = torch.as_tensor(reward, dtype=torch.float)
        done     = torch.as_tensor(done, dtype=torch.float)
        self[self.position] = (state_0, state_1, action, reward, done)
        self.position = (self.position + 1) % self.size

    def sample(self, batch_size, device=None):
        indices = np.random.choice(len(self), batch_size, replace=False)
        states_0 = torch.stack([self.states_0[idx] for idx in indices])
        states_1 = torch.stack([self.states_1[idx] for idx in indices])
        actions = torch.stack([self.actions[idx] for idx in indices])
        rewards = torch.stack([self.rewards[idx] for idx in indices])
        dones = torch.stack([self.dones[idx] for idx in indices])
        if device is not None:
            states_0 = states_0.to(device, non_blocking=True)
            states_1 = states_1.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            rewards = rewards.to(device, non_blocking=True)
            dones = dones.to(device, non_blocking=True)
        return states_0, states_1, actions, rewards, dones

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
            param.data = self.momentum * param.data + (1 - self.momentum) * base_param.data


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

    def discrete_to_cont(self, action):
        action_cont = [(action == 1) * -1 + (action == 2) * 1,  # player x
                    (action == 3) * -1 + (action == 4) * 1,  # player y
                    (action == 5) * -1 + (action == 6) * 1]  # player angle
        return action_cont
    
    def cont_to_discrete(self, action_cont):
        action = torch.zeros(action_cont.shape[0],1,dtype=torch.long, device=action_cont.device, requires_grad=False)
        action[action_cont[:, 0] > 0.5] = 1
        action[action_cont[:, 0] < 0.5] = 2
        action[action_cont[:, 1] > 0.5] = 3
        action[action_cont[:, 1] < 0.5] = 4
        action[action_cont[:, 2] > 0.5] = 5
        action[action_cont[:, 2] < 0.5] = 6
        return action

    def act(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs = normalize_(obs)
        Q = self.module(obs)
        action = self._choose_action(Q)
        self.last_action_discrete = action
        self.last_action = self.discrete_to_cont(action)
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
        for i in range(500):
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
    for _ in range(N):
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


class RLGame:
    def __init__(self, env, max_steps=500):
        self.env = env
        self.max_steps = max_steps
        self.replay_buffer = ReplayBuffer(200000)
        self.num_games = -1
        self.total_steps = 0
        self.reset()

    @property
    def action_agent1(self):
        return self.actions[0]
    
    @action_agent1.setter
    def action_agent1(self, value):
        self.actions[0] = value

    @property
    def action_agent2(self):
        return self.actions[1]
    
    @action_agent2.setter
    def action_agent2(self, value):
        self.actions[1] = value
    
    @property
    def obs_agent1(self):
        return self.observations[0]
    
    @property
    def obs_agent2(self):
        return self.observations[1]

    def reset(self, *args, **kwargs):
        self.num_games += 1
        self.cur_step = 0
        self.observations = [[],[]]
        self.actions = [None, None]
        self.reward = 0.0

        obs1, _ = self.env.reset(*args, **kwargs)
        obs2 = self.env.obs_agent_two()
        self.obs_agent1.append(obs1)
        self.obs_agent2.append(obs2)

    def _stack_state(self, observations, horizon):
        assert len(observations) >= horizon
        return np.stack(observations[-horizon:]) if horizon > 1 else observations[-1]

    def _get_and_set_action(self, agent_idx, agent, horizon):
        observations = self.observations[agent_idx]
        action = self.actions[agent_idx]
        if action is None:
            if len(observations) < horizon:
                action = np.array([0.0,0.0,0.0])
            else:
                agent_inp = self._stack_state(observations, horizon)
                action = agent.act(agent_inp)
            self.actions[agent_idx] = action
        return action

    def _reset_actions(self, action_repeats_agent1=1, action_repeats_agent2=1):
        if self.cur_step % action_repeats_agent1 == 0:
            self.action_agent1 = None
        if self.cur_step % action_repeats_agent2 == 0:
            self.action_agent2 = None

    @torch.no_grad()
    def step(self, agent1, agent2, action_repeats_agent1=1, action_repeats_agent2=1, horizon_agent1=1, horizon_agent2=1):
        self._reset_actions(action_repeats_agent1, action_repeats_agent2)
        action1 = self._get_and_set_action(0, agent1, horizon_agent1)
        action2 = self._get_and_set_action(1, agent2, horizon_agent2)
        action = np.hstack([action1, action2])
        
        obs_agent1, r, done, _, info = self.env.step(action)
        obs_agent2 = self.env.obs_agent_two()

        self.obs_agent1.append(obs_agent1)
        self.obs_agent2.append(obs_agent2)

        state_0 = self._stack_state(self.obs_agent1[:-1], horizon_agent1)
        state_1 = self._stack_state(self.obs_agent1, horizon_agent1)
        self.replay_buffer.add(state_0, state_1, action, r, done)
        self.cur_step += 1
        self.total_steps += 1

        if done or self.cur_step >= self.max_steps:
            self.reset()
            done = True
        return r, done, info

def DQN():
    BS = 64
    GAMMA = 0.995
    LR = 1e-4
    ACTION_REPEATS = 3

    env = lh.LaserHockeyEnv() 
    opponent = lh.BasicOpponent()
    rl_game = RLGame(env)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tracker = Tracker()
    model = MLP([18, 64, 64, 7]).to(device)
    ema_model = EMAModel(MLP([18, 64, 64, 7]).to(device), model, momentum=0.999)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    agent = NNAgent(model, device, action_repeats=ACTION_REPEATS)
    
    #prepopulate(replay_buffer, ACTION_REPEATS)
    #print(len(replay_buffer.buffer))
    
    model.train()
    ema_model.train()
    for game in range(30000):        
        n_updates = 0
        Qs = []
        td_errors = []
        done = False
        while not done:
            _, done, _ = rl_game.step(agent, opponent, action_repeats_agent1=ACTION_REPEATS)
            if len(rl_game.replay_buffer) > 10000 and rl_game.total_steps % 4 == 0:
                n_updates += 1
                optimizer.zero_grad()

                state_0, state_1, action, reward, d = rl_game.replay_buffer.sample(BS, device)
                state_0, state_1 = normalize_(state_0), normalize_(state_1)
                action = agent.cont_to_discrete(action)

                assert action.shape == (BS, 1)
                assert reward.shape == (BS,)
                assert d.shape == (BS,)

                Q = model(state_0)
                Q_next = ema_model(state_1)
                Q_next_max = torch.max(Q_next, dim=1, keepdim=True)[0]
                Q_target = reward[:, None] + GAMMA * Q_next_max * (1 - d[:, None])

                """
                print('action', action.shape)
                print('Q', Q.shape)
                print('Q_next', Q_next.shape)
                print('Q_next_max', Q_next_max.shape)
                print('Q_target', Q_target.shape)
                print('r', reward.shape)
                print('Qgather', Q.gather(1, action).shape)
                print('d', d.shape)
                print('-'*30)"""

                loss = torch.mean((Q_target - Q.gather(1, action)) ** 2)
                loss.backward()
                optimizer.step()
                ema_model.update(model)

                Qs += [Q_target.detach().mean().cpu().item()]
                td_errors += [loss.detach().cpu().item()]
    
        print(f'Game {game}: {n_updates} updates')
        if n_updates > 0:
            tracker.add('value_f', np.mean(Qs), game)
            tracker.add('td_error', np.mean(td_errors), game)
            tracker.add('n_updates', n_updates, game)
    
        if game % 200 == 0 and game > 0:
            print('N Steps:', agent.n)
            evaluate(agent, game, tracker, ACTION_REPEATS, opponent=opponent, N=30)


def main():
    env = lh.LaserHockeyEnv(mode=lh.LaserHockeyEnv.TRAIN_SHOOTING)
    player1 = lh.BasicOpponent()
    player2 = lh.BasicOpponent()
    r, states = play_game(env, player1, player2)
    states[0].save('basic_game.gif', save_all=True, append_images=states[1:], duration=(1/50)*1000, loop=0)

if __name__ == '__main__':
    #main()
    DQN()