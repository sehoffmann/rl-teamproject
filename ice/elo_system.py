from abc import ABC, abstractmethod
import json

import torch

from PIL import Image
import numpy as np

import laserhockey.hockey_env as lh


class Agent(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def act(self, obs):
        pass


class EloLeaderboard(dict):
    """Ideas based on https://en.wikipedia.org/wiki/Elo_rating_system"""

    def __init__(self, start_elo=900, K=32, default_elos=True):
        self.start_elo = start_elo
        self.K = K
        self.elos = {}
        if default_elos:
            self.load_default_ratings()

    def __getitem__(self, key):
        return self.elos[key]

    def __setitem__(self, key, value):
        self.elos[key] = value

    def __contains__(self, key):
        return key in self.elos
    
    def __len__(self):
        return len(self.elos)
    
    def __iter__(self):
        return iter(self.elos)
    
    def __delitem__(self, key):
        del self.elos[key]

    def add_agent(self, agent, elo=None):
        if agent not in self:
            self[agent] = elo if elo is not None else self.start_elo
        else:
            raise ValueError(f"Agent {agent} already in leaderboard")
    
    def get_win_probs(self, agent_a: str, agent_b: str):
        delta = self[agent_b] - self[agent_a]
        p_a_wins = 1 / (1 + 10**(delta / 400))
        return p_a_wins, (1 - p_a_wins)
    
    def calculate_new_elos(self, agent_a: str, agent_b: str, result):
        result_a, result_b = result
        expect_a, expect_b = self.get_win_probs(agent_a, agent_b)
        new_a = self[agent_a] + self.K * (result_a - expect_a)
        new_b = self[agent_b] + self.K * (result_b - expect_b)
        return new_a, new_b

    def update_rating(self, agent_a, agent_b, result):
        """Register the result of the pairing. 
        0 = lost,
        1 = won,
        0.5 = draw
        
        e.g. player 1 beat player 2: (1, 0)
        a draw: (0.5, 0.5)
        """
        self[agent_a], self[agent_b] = self.calculate_new_elos(agent_a, agent_b, result)
        return self[agent_a], self[agent_b]

    def load_default_ratings(self):
        self.elos['basic_weak'] = 885
        self.elos['basic_strong'] = 915

    def save(self, path):
        with open(path, "w") as fp:
            json.dump(self.elos, fp, indent=4)

    def load(self, path):
        with open(path, "r") as fp:
            self.elos = json.load(fp)

    def clone(self):
        new_leaderboard = EloLeaderboard(start_elo=self.start_elo, K=self.K, default_elos=False)
        new_leaderboard.elos = self.elos.copy()
        return new_leaderboard

    def __str__(self):
        sorted_leaderboard = list(sorted(self.elos.items(), key=lambda x: x[1], reverse=True))
        return f"{sorted_leaderboard}"
    
@torch.no_grad()
def play_game(agent1, agent2, max_steps = 250, render=True, action_repeats=1):
    env = lh.HockeyEnv()
    obs_agent1, info = env.reset()
    obs_agent2 = env.obs_agent_two()
    states = []
    result = (0, 0)
    
    if hasattr(agent1, 'reset'):
        agent1.reset()
    if hasattr(agent2, 'reset'):
        agent2.reset()
    
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
            if info['winner'] == 1:
                result = (1, 0)
            elif info['winner'] == -1:
                result = (0, 1)
            else:
                result = (0.5, 0.5)
            if d:
                break
        if d: 
            break
    env.close()
    return r, states, result

class HockeyTournamentEvaluation():
    def __init__(self, add_basics=True, start_elo=900, K=32, default_elos=True):
        self.leaderboard = EloLeaderboard(start_elo=start_elo, K=K, default_elos=default_elos)
        self.agents = {}
        if add_basics:
            self.agents['basic_weak'] = lh.BasicOpponent()
            self.agents['basic_strong'] = lh.BasicOpponent(weak=False)
    
    def __getitem__(self, key):
        return self.agents[key]

    def __setitem__(self, key, value):
        self.agents[key] = value

    def __delitem__(self, key):
        del self.agents[key]

    def __contains__(self, key):
        return key in self.agents

    def __len__(self):
        return len(self.agents)
    
    def __iter__(self):
        return iter(self.agents)
    
    def add_agent(self, name, agent, elo=None, num_games=0):
        if name in self.agents:
            raise ValueError("Agent already registered")
        self.agents[name] = agent
        self.leaderboard.add_agent(name, elo=elo)
        if num_games > 0:
            self.evaluate_agent(name, agent, n_games=num_games)
    
    def get_pairing(self, name):
        alpha = 0.5 # softness factor
        elo = self.leaderboard[name]
        sample_weights = [np.abs(elo - opp_elo)**alpha for opp_elo in self.leaderboard.elos.values()]        
        sample_weights = np.array(sample_weights) / sum(sample_weights)
        # print(sample_weights)
        opponents = np.random.choice(list(self.leaderboard.elos.keys()), size=1, p=sample_weights)
        return (name, opponents[0])
    
    def random_plays(self, n_plays=10, verbose=False):
        names = list(self.agents.keys())
        for _ in range(n_plays):
            name1 = np.random.choice(names)
            _, name2 = self.get_pairing(name1) 
            _, _, res = play_game(self[name1], self[name2])
            new_elo1, new_elo2 = self.leaderboard.update_rating(name1, name2, result=res)
            if verbose:
                print(f'{name1} ({new_elo1:.1f}) vs {name2} ({new_elo2:.1f}): {res}')

    def run_n_games(self, name1, name2, n = 1, save_gif=False):
        for _ in range(n):
            _, states, res = play_game(self[name1], self[name2], render=save_gif)
            _ = self.leaderboard.update_rating(name1, name2, result=res)
        
        if save_gif:
            states[0].save(f'tournament_{name1}_{name2}.gif', save_all=True, append_images=states[1:], duration=(1/50)*1000)

    def evaluate_agent(self, name, agent, n_games=1, verbose=False):
        assert name in self, "register your agent for this tournament to receive elo"
        for _ in range(n_games):
            # note: n games varying pairings
            (name_agent_1, name_agent_2) = self.get_pairing(name)
            if verbose:
                print(f"paring: {name_agent_1}, {name_agent_2}")
            self.run_n_games(name_agent_1, name_agent_2, n=1)
        return self.leaderboard[name]

    def __str__(self):
        return self.leaderboard.__str__()


if __name__ == "__main__":

    # get baselines for basic opponents
    tournament = HockeyTournamentEvaluation(restart=True)

    tournament.random_plays(n_plays=1000)

    print(tournament)
