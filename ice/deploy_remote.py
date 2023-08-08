import argparse
from tournament_client.client.remoteControllerInterface import RemoteControllerInterface
from dqn import NNAgent
from numpy import ndarray
from pathlib import Path
import numpy as np
from tournament_client.client.backend.client import Client


class RemoteNNAgent(RemoteControllerInterface):
    def __init__(self, identifier: str, checkpoint_loc: str) -> None:
        super().__init__(identifier)
        
        self.agent = NNAgent.load_model(checkpoint_loc, device='cpu')

    def remote_act(self, obs: ndarray) -> ndarray:
        action = np.array(self.agent.act(obs))
        return action

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

    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--identifier', type=str, default='iceq')
    parser.add_argument('--username', type=str, default='user0')
    parser.add_argument('--password', type=str, default='1234')
    parser.add_argument('--num_games', type=int, default=None)


    args = parser.parse_args()

    controller = RemoteNNAgent(args.identifier, args.checkpoint)

    client = create_client(controller, args)

    
    

