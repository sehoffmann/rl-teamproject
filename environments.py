import numpy as np

import laserhockey.laser_hockey_env as lh   
from gymnasium.spaces import Box, Discrete

N_DISCRETE_ACTIONS = 7
ACT_DIM = 3
OBS_DIM = 18

class LaserHockeyWithOpponent(lh.LaserHockeyEnv):
    """A Wrapper for the Laserhockey environment but with a specified opponent.
    
    Action-Space: 3-dimensional vector
    * Bounded to [-1,1]
    * Target-Pos X
    * Target-Pos Y
    * Target-Angle
    """
    def __init__(self, opponent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opponent = opponent
        # FIXME: dytpes are not consistent...
        self.action_space = Box(low=-1, high=1, shape=(ACT_DIM,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float64)

    def step(self, action):
        if self.mode == self.TRAIN_DEFENSE or self.mode == self.TRAIN_SHOOTING:
            opponent_action = np.zeros_like(action)
        else:
            opponent_obs = self.obs_agent_two()
            opponent_action = self.opponent.act(opponent_obs)
        
        return super().step(np.hstack([action, opponent_action]))

class LaserHockeyWithOpponentAndDiscreteActions(LaserHockeyWithOpponent):
    """Wrapper that allows discrete actions."""
    def __init__(self, opponent, *args, **kwargs):
        super().__init__(opponent, *args, **kwargs)
        # TODO: n_actions
        self.action_space = Discrete(N_DISCRETE_ACTIONS)

    def step(self, action):
        cont_action = self.discrete_to_continous_action(action)
        return super().step(cont_action)