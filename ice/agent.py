import copy
import json
from pathlib import Path

import torch
import numpy as np

from environments import IcyHockey
from replay_buffer import FrameStacker

class NNAgent:
    ENV = IcyHockey()

    def __init__(self, model, device, frame_stacks=None):
        self.model = model
        self.device = device
        self.stacker = FrameStacker(frame_stacks if frame_stacks is not None else 1)

    def reset(self):
        self.stacker.clear()

    def act(self, state):
        state = self.stacker.append_and_stack(state)
        action_discrete = self.select_action(state, train=False)
        return self.ENV.discrete_to_continous_action(action_discrete)

    def copy(self, eval=True):
        model = copy.deepcopy(self.model)
        model.requires_grad_(False)
        if eval:
            model.eval().requires_grad_(False)
        return NNAgent(model, self.device, self.stacker.num_frames)

    def save_model(self, path):
        self.model.to('cpu')
        torch.save(self.model, path)
        self.model.to(self.device)

    @classmethod
    def _load_model(cls, model, config, device):
        raise NotImplementedError()

    @classmethod
    def load_model(cls, path, device):
        path = Path(path)
        try:
            with open(path.parent / f'{path.stem}.json', 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            config = None

        if config is None:
            with open(path.parent / 'config.json', 'r') as f:
                config = json.load(f)

        model = torch.load(path, map_location=device)
        model.eval().requires_grad_(False)
        if config.get('class', 'DQN') == 'DQN':
            from dqn import DqnInferenceAgent
            return DqnInferenceAgent._load_model(model, config, device)
        else:
            raise ValueError(f"Unknown agent class: {config['class']}")

    @classmethod
    def load_lilith_weak(cls, device):
        path = Path('baselines') / 'lilith_weak.pt'
        return cls.load_model(path, device)


class MajorityVoteAgent:
    ENV = IcyHockey()

    def __init__(self, agents):
        self.agents = agents

    def act(self, obs):
        action_discrete = self.select_action(obs)
        return self.ENV.discrete_to_continous_action(action_discrete)

    def select_action(self, obs):
        votes = []
        for agent in self.agents:
            state = agent.stacker.append_and_stack(obs)
            action_discrete = agent.select_action(state)
            votes.append(action_discrete)
        action, counts = np.unique(votes, return_counts=True)
        return action[np.argmax(counts)]

    def reset(self):
        for agent in self.agents:
            if hasattr(agent, 'reset'):
                agent.reset()