import argparse
import pprint
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from elo_system import HockeyTournamentEvaluation
from environments import IcyHockey
from agent import MajorityVoteAgent, NNAgent
import numpy as np
from plotting import save_games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('-o', '--out', type=Path, default='self-play.gif')
    parser.add_argument('-e', '--ensemble', type=int, default=0)
    parser.add_argument('-n', '--games', type=int, default=10)
    parser.add_argument('-s', '--speed', type=int, default=1000)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_dir = args.checkpoint.parent
    frame = int(args.checkpoint.stem.split('_')[-1])
    agents = []
    for i in range(args.ensemble + 1):
        cp = model_dir / f'frame_{frame - 100_000 * i:010d}.pt'
        agents.append(NNAgent.load_model(cp, device=device))
    agent1 = MajorityVoteAgent(agents)
    agent2 = agent1.copy()

    env = IcyHockey()
    game_imgs = env.rollout(agent1, agent2, args.games)
    save_games(args.out, game_imgs, speed=args.speed)

if __name__ == '__main__':
    main()