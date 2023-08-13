import argparse
import pprint
from pathlib import Path
import torch
from elo_system import HockeyTournamentEvaluation
from dqn import NNAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints', nargs='*', default=[])
    parser.add_argument('-s', '--start', type=int, default=200_000)
    parser.add_argument('-f', '--freq', type=int, default=200_000)
    parser.add_argument('-n', '--games', type=int, default=5000)
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('--no-basics', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tournament = HockeyTournamentEvaluation(add_basics=not args.no_basics, default_elos=False)
    for path in args.checkpoints:
        path = Path(path)
        for frame_idx in range(args.start, 30_000_000, args.freq):
            cp = path / f'frame_{frame_idx:010d}.pt'
            if not cp.exists():
                break
            tournament.add_agent(f'{path.name}_{frame_idx:010d}', NNAgent.load_model(cp, device=device))
    
    print(f'Playing {args.games} games...')
    i = 1
    for _ in tournament.random_plays(n_plays=args.games, verbose=not args.quiet):
        if i % 500 == 0:
            print(f'Played {i} games')
            elos = tournament.leaderboard.mean_elos()
            elos_sorted = list(sorted(elos.items(), key=lambda x: x[1], reverse=True))
            pprint.pprint(elos_sorted, indent=4)
            print()
            tournament.leaderboard.save(f'evaluation.json')  
        i += 1

if __name__ == '__main__':
    main()