import argparse
import torch
from tournament_client.client.remoteControllerInterface import RemoteControllerInterface
from dqn import NNAgent
from numpy import ndarray
from pathlib import Path
import numpy as np
from tournament_client.client.backend.client import Client
from environments import IcyHockey

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

    def before_game_starts(self):
        for agent in self.agents:
            if hasattr(agent, 'reset'):
                agent.reset()


class RemoteNNAgent(RemoteControllerInterface):
    def __init__(self, identifier, agent):
        super().__init__(identifier)
        self.agent = agent

    def remote_act(self, obs: ndarray) -> ndarray:
        action = np.array(self.agent.act(obs))
        return action
    
    def before_game_starts(self):
        if hasattr(self.agent, 'reset'):
            self.agent.reset()

get_log_dir = lambda name: f"tournament_client/logs/{name}"

def create_client(controller: RemoteControllerInterface, args):
    client = Client(username=args.username,
                    password=args.password,
                    controller=controller,
                    output_path=get_log_dir(args.identifier), # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    # server_addr='localhost',
                    num_games=args.num_games)
    return client

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('checkpoints', nargs='+')
    parser.add_argument('--identifier', type=str, default='iceq')
    parser.add_argument('--username', type=str, default='user0')
    parser.add_argument('--password', type=str, default='1234')
    parser.add_argument('--num-games', type=int, default=None)
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

    agents = [NNAgent.load_model(cp, device=device) for cp in args.checkpoints]
    ensemble = MajorityVoteAgent(agents)

    controller = RemoteNNAgent(args.identifier, ensemble)
    client = create_client(controller, args)

    
    

