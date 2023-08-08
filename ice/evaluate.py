import argparse
import torch
from elo_system import HockeyTournamentEvaluation
from dqn import NNAgent
from dqn_stenz import get_stenz

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints', nargs='*', default=[])
    parser.add_argument('-n', '--games', type=int, default=1000)
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('--no-basics', action='store_true')
    parser.add_argument('--no-default', action='store_true')
    parser.add_argument('--no-stenz', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tournament = HockeyTournamentEvaluation(add_basics=not args.no_basics, default_elos=not args.no_default)
    for cp in args.checkpoints:
        tournament.add_agent(cp, NNAgent.load_model(cp, device=device))
    
    if not args.no_stenz:
        tournament.add_agent('stenz', get_stenz())

    print(f'Playing {args.games} games...')
    try:
        tournament.random_plays(n_plays=args.games, verbose=not args.quiet)
    except KeyboardInterrupt:
        pass
    print(tournament)

if __name__ == '__main__':
    main()