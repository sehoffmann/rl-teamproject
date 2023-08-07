from abc import ABC, abstractmethod

import pickle
import itertools
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
    K = 32
    R_INIT = 900
    SAVE_PATH = "elo_leaderboard.pkl"
    def __init__(self, load_stored= False):

        if load_stored:
            self._load_leaderboard()
        else:
            ## these values are determined by letting the two play 1000 times against each other
            self.elo_system = {
                'basic_weak': 885,
                'basic_strong': 915
            }

    def _load_leaderboard(self):
        try:
            with open(EloLeaderboard.SAVE_PATH, "rb") as fp:
                self.elo_system = pickle.load(file=fp)
        except FileNotFoundError:
            print("creating new leaderboard since not exists")
            self.elo_system = {}

    def _store_leaderboard(self):
        with open(EloLeaderboard.SAVE_PATH, "wb") as fp:
            pickle.dump(self.elo_system, file=fp)

    def get_elo_score(self, agent_name: str):
        if agent_name not in self.elo_system:
            self.elo_system[agent_name] = EloLeaderboard.R_INIT
        return self.elo_system[agent_name]
    
    def set_elo_score(self, agent_name: str, value: int):
        self.elo_system[agent_name] = value
    
    def get_win_probs(self, agent_a: str, agent_b: str):
        rating_a = self.get_elo_score(agent_a)
        rating_b = self.get_elo_score(agent_b)
        delta = rating_b - rating_a
        p_a_wins = 1 / (1 + 10**(delta / 400))
        return p_a_wins, (1 - p_a_wins)
    
    def update_rating(self, pairing = ("agent_a", "agent_b"), result = (0, 0), save=True):
        """Register the result of the pairing. 
        0 = lost,
        1 = won,
        0.5 = draw
        
        e.g. player 1 beat player 2: (1, 0)
        a draw: (0.5, 0.5)
        """
        agent_a, agent_b = pairing
        result_a, result_b = result
        rating_a = self.get_elo_score(agent_a)
        rating_b = self.get_elo_score(agent_b)
        expect_a, expect_b = self.get_win_probs(agent_a, agent_b)
        new_a = rating_a + EloLeaderboard.K * (result_a - expect_a)
        new_b = rating_b + EloLeaderboard.K * (result_b - expect_b)

        if save:
            self.set_elo_score(agent_a, value=new_a)
            self.set_elo_score(agent_b, value=new_b)
            self._store_leaderboard()

        return new_a, new_b

    def __str__(self):
        sorted_leaderboard = list(sorted(self.elo_system.items(), key=lambda x: x[1], reverse=True))
        return f"{sorted_leaderboard}"
    
@torch.no_grad()
def play_game(agent1, agent2, max_steps = 250, render=True, action_repeats=1):
    env = lh.HockeyEnv()
    obs_agent1, info = env.reset()
    obs_agent2 = env.obs_agent_two()
    states = []
    result = (0, 0)
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
    def __init__(self, restart=False):
        """If restart = True, it starts with only the weak and strong opp"""
        self.elo_leaderboard = EloLeaderboard(load_stored=not restart)
        self.agent_register = {
            'basic_weak': lh.BasicOpponent(),
            'basic_strong': lh.BasicOpponent(weak=False)
        }
    
    def get_agent_instance(self, agent_name):
        try:
            return self.agent_register[agent_name]
        except KeyError:
            raise Error("Agent not registered, use register function")

    def register_agent(self, agent_name: str, agent, update=False):
        """Call this once to add your agent to the rating system
        you can also call it again to update the instance associated with the given agent name.
        Note: when you register a new agent, it gets evaluated once.
        """
        self.agent_register[agent_name] = agent
        if not update:
            self.evaluate_agent(agent_name, agent, n_games=1)

    def is_registered(self, agent_name):
        return agent_name in self.agent_register.keys()
    
    def get_pairing(self, agent_name):
        alpha = 0.5 # softness factor
        agent_elo = self.elo_leaderboard.get_elo_score(agent_name)
        sample_weights = [np.abs(agent_elo - opp_elo)**alpha for opp_elo in self.elo_leaderboard.elo_system.values()]        
        sample_weights = np.array(sample_weights) / sum(sample_weights)
        # print(sample_weights)
        opponents = np.random.choice(list(self.elo_leaderboard.elo_system.keys()), size=1, p=sample_weights)
        return (agent_name, opponents[0])
    
    def random_plays(self, n_plays=10):
        for (name, agent) in self.agent_register.items():
            for _ in range(n_plays):
                (name_1, name_2) = self.get_pairing(name)
                self.run_n_games(name_1, name_2, agent, self.agent_register[name_2])

    def run_n_games(self, agent_1_name, agent_2_name, agent_1, agent_2, n = 1, save_gif=False):
        """Note: n games SAME pairing"""
        for _ in range(n):
            _, states, res = play_game(agent_1, agent_2, render=False)
            _ = self.elo_leaderboard.update_rating(pairing=(agent_1_name, agent_2_name), result=res)
        # print(res)
        if save_gif:
            states[0].save(f'tournament_{agent_1_name}_{agent_2_name}.gif', save_all=True, append_images=states[1:], duration=(1/50)*1000)

    def evaluate_agent(self, agent_name, agent, n_games=1, verbose=False):
        assert self.is_registered(agent_name), "register your agent for this tournament to receive elo"
        for _ in range(n_games):
            # note: n games varying pairings
            (name_agent_1, name_agent_2) = self.get_pairing(agent_name)
            if verbose:
                print(f"paring: {name_agent_1}, {name_agent_2}")
            agent_1_instance = agent
            agent_2_instance = self.get_agent_instance(name_agent_2)
            self.run_n_games(name_agent_1, name_agent_2, agent_1_instance, agent_2_instance, n=1)
        return self.elo_leaderboard.get_elo_score(agent_name)

    def __str__(self):
        return self.elo_leaderboard.__str__()


if __name__ == "__main__":

    # get baselines for basic opponents
    tournament = HockeyTournamentEvaluation(restart=True)

    tournament.random_plays(n_plays=1000)

    print(tournament)