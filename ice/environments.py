from laserhockey.hockey_env import HockeyEnv_BasicOpponent
from gymnasium.spaces import Box, Discrete

class DiscreteHockey_BasicOpponent(HockeyEnv_BasicOpponent):
    N_DISCRETE_ACTIONS = 7
    """Wrapper that allows discrete actions."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = Discrete(self.N_DISCRETE_ACTIONS)

    def reset(self, *args, **kwargs):
        state, info = super().reset()
        return state

    def step(self, action):
        cont_action = self.discrete_to_continous_action(action)
        state, reward, done, truncated, info = super().step(cont_action)
        if False:
            return state, reward, done, truncated
        else:
            return state, reward, done, truncated, info   