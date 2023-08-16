import argparse
import pprint
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from elo_system import HockeyTournamentEvaluation
from agent import MajorityVoteAgent, NNAgent
import numpy as np

def save(tournament, agent_names):
    elos = tournament.leaderboard.mean_elos()
    elos_sorted = list(sorted(elos.items(), key=lambda x: x[1], reverse=True))
    pprint.pprint(elos_sorted, indent=4)
    print()
    tournament.leaderboard.save(f'evaluation2.json')

    plt.figure(figsize=(12, 8))
    for agent_name in agent_names:
        agent_elos = [elo for name, elo in elos.items() if name.startswith(agent_name)]
        agent_frames = [int(name.split('_')[-1]) for name in elos if name.startswith(agent_name)]
        agent_elos = np.convolve(np.pad(agent_elos, 1, mode='edge'), np.ones(3)/3, mode='valid') 
        plt.plot(agent_frames, agent_elos, label=agent_name)
    
    plt.axhline(tournament.leaderboard['basic_weak'], linestyle='--', label='basic_weak')
    plt.axhline(tournament.leaderboard['basic_strong'], linestyle='--', label='basic_strong')
    
    plt.xlabel('Frames')
    plt.title('ELO')
    plt.legend()


    plt.savefig(f'evaluation2.png', bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints', nargs='*', default=[])
    parser.add_argument('-s', '--start', type=int, default=200_000)
    parser.add_argument('-f', '--freq', type=int, default=200_000)
    parser.add_argument('-e', '--ensemble', type=int, default=0)
    parser.add_argument('-n', '--games', type=int, default=5000)
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('--no-basics', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tournament = HockeyTournamentEvaluation(add_basics=not args.no_basics, default_elos=False)
    agents = {}
    for path in args.checkpoints:
        path = Path(path)
        name = path.name
        agents[name] = []
        for frame_idx in range(args.start, 30_000_000, args.freq):
            cp = path / f'frame_{frame_idx:010d}.pt'
            if not cp.exists():
                break
            agent = MajorityVoteAgent([NNAgent.load_model( path / f'frame_{frame_idx - 100_000 * i:010d}.pt', device=device) for i in range(args.ensemble + 1)])
            agent_name = f'{path.name}_{frame_idx:010d}'
            tournament.add_agent(agent_name, agent)
            agents[name].append(agent_name)
 
    print('Warming up...')
    prev_opponents = ['basic_weak', 'basic_strong']
    for _ in tournament.random_plays(n_plays=20, agents=prev_opponents, opponents=prev_opponents, verbose=False):
        pass
    for cp, agent_names in agents.items():
        print(cp)
        for _ in tournament.random_plays(n_plays=100, agents=agent_names, opponents=prev_opponents, verbose=False, update_opponents=False):
            pass
        for _ in tournament.random_plays(n_plays=100, agents=agent_names, opponents=list(agent_names), verbose=False):
            pass
        prev_opponents += list(agent_names)
    for _ in tournament.random_plays(n_plays=20, agents=['basic_weak', 'basic_strong'], update_opponents=False, verbose=False):
        pass

    print(f'Playing {args.games} games...')
    i = 1 
    for _ in tournament.random_plays(n_plays=args.games, verbose=not args.quiet):
        if i % 100 == 0:
            print(f'Played {i} games')
            save(tournament, agents)
        i += 1
    save(tournament, agents)

if __name__ == '__main__':
    main()