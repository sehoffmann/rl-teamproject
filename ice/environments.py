from collections import deque
import numpy as np

from laserhockey.hockey_env import HockeyEnv, BasicOpponent
import gymnasium.spaces as spaces


class Opponent:

    def __init__(self, agent, prob):
        self.agent = agent
        self.prob = prob


class RollingOpponent(Opponent):

    def __init__(self, num_rolls):
        self.num_rolls = num_rolls
        self.opps = deque(maxlen=num_rolls)
        self.idx = 0

    def add_opponent(self, agent, prob):
        self.opps.append(Opponent(agent, prob))

    @property
    def agent(self):
        p = np.array([opp.prob for opp in self.opps], dtype=np.float64)
        p /= p.sum()
        return np.random.choice(self.opps, p=p).agent

    @property
    def prob(self):
        return sum([opp.prob for opp in self.opps])


class IcyHockey(HockeyEnv):
    N_DISCRETE_ACTIONS = 7

    def __init__(self, mode=HockeyEnv.NORMAL, reward_shaping=True):
        # this has to be done before calling super().__init__(), because reset() is called
        self.opponents = {}
        self.reward_shaping = reward_shaping
        self.cur_opp_name = None
        self.cur_opp_agent = None
        self.add_basic_opponent(weak=True)
        super().__init__(mode=mode, keep_mode=True)
        self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS)

    def add_opponent(self, name, agent, prob, rolling=1):
        if rolling > 1:
            if name not in self.opponents:
                self.opponents[name] = RollingOpponent(rolling)
            self.opponents[name].add_opponent(agent, prob)
        else:
            self.opponents[name] = Opponent(agent, prob)

    def add_basic_opponent(self, weak, prob=None):
        if weak:
            prob = 1.0 if prob is None else prob
            self.add_opponent('basic_weak', BasicOpponent(weak=True), prob)
        else:
            prob = 4.0 if prob is None else prob
            self.add_opponent('basic_strong', BasicOpponent(weak=False), prob)

    def remove_opponent(self, name):
        del self.opponents[name]

    def has_opponent(self, name):
        return name in self.opponents

    def sample_opponent(self):
        probs = np.array([opp.prob for opp in self.opponents.values()], dtype=np.float64)
        probs /= probs.sum()
        name = np.random.choice(list(self.opponents.keys()), p=probs)
        agent = self.opponents[name].agent
        return name, agent

    def reset(self, *args, **kwargs):
        self.cur_opp_name, self.cur_opp_agent = self.sample_opponent()
        if hasattr(self.cur_opp_agent, 'reset'):
            self.cur_opp_agent.reset()
        obs, info = super().reset()
        self._augment_info(info)
        return obs, info

    def step(self, action):
        a1 = self.discrete_to_continous_action(action)
        ob2 = self.obs_agent_two()
        a2 = self.cur_opp_agent.act(ob2)

        stacked_action = np.hstack([a1, a2])
        obs, reward, done, _, info = super().step(stacked_action)
        self._augment_info(info)
        if not self.reward_shaping:
            reward = info['winner'] * 10

        return obs, reward, done, _, info
    
    def _augment_info(self, info):
        info['opponent'] = self.cur_opp_name
        return info