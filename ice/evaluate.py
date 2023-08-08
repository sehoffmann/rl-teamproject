import argparse
import torch
from elo_system import HockeyTournamentEvaluation
from dqn import NNAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints', nargs='*', default=[])
    parser.add_argument('-n', '--games', type=int, default=1000)
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('--no-basics', action='store_true')
    parser.add_argument('--no-default', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tournament = HockeyTournamentEvaluation(add_basics=not args.no_basics, default_elos=not args.no_default)
    for cp in args.checkpoints:
        tournament.add_agent(cp, NNAgent.load_model(cp, device=device))
    
    print(f'Playing {args.games} games...')
    tournament.random_plays(n_plays=args.games, verbose=not args.quiet)
    print(tournament)

if __name__ == '__main__':
    main()