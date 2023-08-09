import argparse
import torch
from elo_system import HockeyTournamentEvaluation
from dqn import NNAgent
from dqn_stenz import get_stenz
import pandas as pd
from deploy_remote import MajorityVoteAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints', nargs='*', default=[])
    parser.add_argument('-n', '--games', type=int, default=5000)
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
        tournament.add_agent('stenz1', get_stenz('baselines/stenz.pth', device=device))
        tournament.add_agent('stenz2', get_stenz('baselines/stenz_29700.pth', device=device))
        tournament.add_agent('stenz3', get_stenz('baselines/stenz_37400.pth', device=device))

    ensemble1 = MajorityVoteAgent([
        NNAgent.load_model('models/final-BBLN_20230809_03:16/frame_0004500000.pt', device=device),
        NNAgent.load_model('models/final-BBLN_20230809_03:16/frame_0004600000.pt', device=device),
        NNAgent.load_model('models/final-BBLN_20230809_03:16/frame_0004700000.pt', device=device),
        NNAgent.load_model('models/final-BBLN_20230809_03:16/frame_0004800000.pt', device=device),
        NNAgent.load_model('models/final-BBLN_20230809_03:16/frame_0004900000.pt', device=device),
        NNAgent.load_model('models/final-BBLN_20230809_03:16/frame_0005000000.pt', device=device),
    ])
    tournament.add_agent('ensemble_bbln_temporal', ensemble1)

    ensemble2 = MajorityVoteAgent([ 
        NNAgent.load_model('models/final-BBLN-crps_20230809_03:17/frame_0004500000.pt', device=device),
        NNAgent.load_model('models/final-BBLN-crps_20230809_03:17/frame_0004400000.pt', device=device),
        NNAgent.load_model('models/final-BBLN-crps_20230809_03:17/frame_0004300000.pt', device=device),
        NNAgent.load_model('models/final-BBLN-crps_20230809_03:17/frame_0004200000.pt', device=device),
        NNAgent.load_model('models/final-BBLN-crps_20230809_03:17/frame_0004100000.pt', device=device),
        NNAgent.load_model('models/final-BBLN-crps_20230809_03:17/frame_0004000000.pt', device=device),
        NNAgent.load_model('models/final-BBLN-crps_20230809_03:17/frame_0003900000.pt', device=device),
        NNAgent.load_model('models/final-BBLN-crps_20230809_03:17/frame_0003800000.pt', device=device),
        NNAgent.load_model('models/final-BBLN-crps_20230809_03:17/frame_0003700000.pt', device=device),
    ])
    tournament.add_agent('ensemble_bbln_crps_temporal', ensemble2)

    ensemble3 = MajorityVoteAgent([
        NNAgent.load_model('models/final-BBLN-crps_20230809_03:17/frame_0004100000.pt', device=device),
        NNAgent.load_model('models/final-BBLN-crps_20230809_03:17/frame_0004000000.pt', device=device),
        NNAgent.load_model('models/final-baseline1-LN_20230809_03:19/frame_0006000000.pt', device=device),
        NNAgent.load_model('models/final-BBLN-crps-explore-adv-phase2_20230809_06:30/frame_0002000000.pt', device=device),
        NNAgent.load_model('models/final-baseline1-LN-stacked_20230809_03:19/frame_0006100000.pt', device=device),
        NNAgent.load_model('models/final-BBLN-DoubleQ_20230809_03:20/frame_0004300000.pt', device=device),
    ])
    tournament.add_agent('rainbow_ensemble', ensemble3)

    print(f'Playing {args.games} games...')
    try:
        elos = []
        for cur_elos in tournament.random_plays(n_plays=args.games, verbose=not args.quiet):
            elos.append(cur_elos)
    except KeyboardInterrupt:
        pass
    df = pd.DataFrame.from_records(elos)
    print('--------------------------')
    print()
    print(df.tail(400).mean().sort_values(ascending=False))

if __name__ == '__main__':
    main()