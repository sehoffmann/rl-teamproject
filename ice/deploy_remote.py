import argparse
import torch
import os
from dqn import NNAgent
from numpy import ndarray
from pathlib import Path
import numpy as np
from client.remoteControllerInterface import RemoteControllerInterface
from client.backend.client import Client
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

def create_client(controller: RemoteControllerInterface, username, password, identifier):
    out_dir = Path(os.environ.get('SCRATCH','')) / 'tournament_client' / identifier 
    client = Client(username=username,
                    password=password,
                    controller=controller,
                    output_path=str(out_dir), # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    # server_addr='localhost',
                    num_games=args.num_games)
    return client

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--preset', type=str)
    parser.add_argument('--checkpoints', nargs='*')
    parser.add_argument('--identifier', type=str, default='iceq')
    parser.add_argument('--num-games', type=int, default=None)
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

    username = 'Tübingen Ice Q-Learners'
    password = 'fie9Amai6r'

    if args.preset == 'crps':
        dir1 = 'models/final-BBLN-crps_20230809_03:17/'
        dir2 = 'models/final-BBLN-crps-explore_20230809_03:17/'
        ensemble = MajorityVoteAgent([
            NNAgent.load_model(dir1 + 'frame_0007500000.pt', device=device, ucb=True),
            NNAgent.load_model(dir1 + 'frame_0007600000.pt', device=device, ucb=True),
            NNAgent.load_model(dir1 + 'frame_0007700000.pt', device=device, ucb=True),
        ])
        identifier = 'crps'
    elif args.preset == 'adv':
        dir1 = 'models/final-BBLN-crps-explore-adv-RERUN_20230809_11:10/'
        dir2 = 'models/final-BBLN-crps-explore-adv-RERUN_20230809_11:15/'
        agents = [
            NNAgent.load_model(dir1 + 'frame_0003600000.pt', device=device),
            NNAgent.load_model(dir1 + 'frame_0003700000.pt', device=device),
            NNAgent.load_model(dir2 + 'frame_0003600000.pt', device=device),
            NNAgent.load_model(dir2 + 'frame_0003700000.pt', device=device),
        ]
        ensemble = MajorityVoteAgent(agents)
        identifier = 'crps-adv-explore'
    else:
        agents = [NNAgent.load_model(cp, device=device) for cp in args.checkpoints]
        ensemble = MajorityVoteAgent(agents)
        identifier = args.identifier

    controller = RemoteNNAgent(identifier, ensemble)
    client = create_client(controller, username, password, identifier)

    
    

