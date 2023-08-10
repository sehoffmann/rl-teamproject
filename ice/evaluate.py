import argparse
import pprint
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
    parser.add_argument('--no-stenz', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tournament = HockeyTournamentEvaluation(add_basics=not args.no_basics, default_elos=False)
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

    if True:
        dir1 = '/mnt/qb/work2/goswami0/gkd021/code/rl-teamproject/models/final-BBLN-crps-explore_20230809_03:17/'
        dir2 = '/mnt/qb/work2/goswami0/gkd021/code/rl-teamproject/models/final-BBLN-crps-explore_20230809_03:45/'
        dir3 = '/mnt/qb/work2/goswami0/gkd021/code/rl-teamproject/models/final-BBLN-crps_20230809_03:17/'
        agents = [
            NNAgent.load_model(dir1 + 'frame_0004300000.pt', device=device),
            NNAgent.load_model(dir1 + 'frame_0004400000.pt', device=device),
            NNAgent.load_model(dir1 + 'frame_0004500000.pt', device=device),
            NNAgent.load_model(dir1 + 'frame_0004600000.pt', device=device),
            NNAgent.load_model(dir1 + 'frame_0004700000.pt', device=device),
            NNAgent.load_model(dir1 + 'frame_0004800000.pt', device=device),
            NNAgent.load_model(dir2 + 'frame_0003800000.pt', device=device),
            NNAgent.load_model(dir2 + 'frame_0003900000.pt', device=device),
            NNAgent.load_model(dir3 + 'frame_0004900000.pt', device=device),
            NNAgent.load_model(dir3 + 'frame_0005000000.pt', device=device),
            NNAgent.load_model(dir3 + 'frame_0005100000.pt', device=device),
        ]
        ensemble = MajorityVoteAgent(agents)
        tournament.add_agent('LIVE-Version1', ensemble)

        dir1 = 'models/final-BBLN-crps_20230809_03:17/'
        dir2 = 'models/final-BBLN-crps-explore_20230809_03:17/'
        ensemble = MajorityVoteAgent([
            NNAgent.load_model(dir1 + 'frame_0005900000.pt', device=device),
            NNAgent.load_model(dir1 + 'frame_0006000000.pt', device=device),
            NNAgent.load_model(dir1 + 'frame_0006100000.pt', device=device),
            NNAgent.load_model(dir1 + 'frame_0006200000.pt', device=device),
            NNAgent.load_model(dir2 + 'frame_0005900000.pt', device=device),
            NNAgent.load_model(dir2 + 'frame_0006000000.pt', device=device),
            NNAgent.load_model(dir2 + 'frame_0006100000.pt', device=device),
            NNAgent.load_model(dir2 + 'frame_0006200000.pt', device=device),
        ])
        tournament.add_agent('LIVE-Version2', ensemble)

        dir1 = 'models/final-BBLN-crps_20230809_03:17/'
        dir2 = 'models/final-BBLN-crps-explore_20230809_03:17/'
        ensemble = MajorityVoteAgent([
            NNAgent.load_model(dir1 + 'frame_0006500000.pt', device=device, ucb=True),
            NNAgent.load_model(dir1 + 'frame_0006600000.pt', device=device, ucb=True),
            NNAgent.load_model(dir1 + 'frame_0006700000.pt', device=device, ucb=True),
            NNAgent.load_model(dir1 + 'frame_0006800000.pt', device=device, ucb=True),
            NNAgent.load_model(dir2 + 'frame_0006500000.pt', device=device, ucb=True),
            NNAgent.load_model(dir2 + 'frame_0006600000.pt', device=device, ucb=True),
            NNAgent.load_model(dir2 + 'frame_0006700000.pt', device=device, ucb=True),
            NNAgent.load_model(dir2 + 'frame_0006800000.pt', device=device, ucb=True),
        ])
        tournament.add_agent('LIVE-Version3', ensemble)

        dir1 = 'models/final-BBLN-crps_20230809_03:17/'
        dir2 = 'models/final-BBLN-crps-explore_20230809_03:17/'
        ensemble = MajorityVoteAgent([
            NNAgent.load_model(dir1 + 'frame_0007500000.pt', device=device, ucb=True),
            NNAgent.load_model(dir1 + 'frame_0007600000.pt', device=device, ucb=True),
            NNAgent.load_model(dir1 + 'frame_0007700000.pt', device=device, ucb=True),
            #NNAgent.load_model(dir2 + 'frame_0007500000.pt', device=device, ucb=True),
            #NNAgent.load_model(dir2 + 'frame_0007600000.pt', device=device, ucb=True),
            #NNAgent.load_model(dir2 + 'frame_0007700000.pt', device=device, ucb=True),
        ])
        tournament.add_agent('LIVE-Version4', ensemble)

        dir1 = 'models/final-BBLN-crps_20230809_03:17/'
        dir2 = 'models/final-BBLN-crps-explore_20230809_03:17/'
        ensemble = MajorityVoteAgent([
            NNAgent.load_model(dir1 + 'frame_0011400000.pt', device=device, ucb=True),
            NNAgent.load_model(dir1 + 'frame_0011500000.pt', device=device, ucb=True),
            NNAgent.load_model(dir1 + 'frame_0011600000.pt', device=device, ucb=True),
            #NNAgent.load_model(dir2 + 'frame_0007500000.pt', device=device, ucb=True),
            #NNAgent.load_model(dir2 + 'frame_0007600000.pt', device=device, ucb=True),
            #NNAgent.load_model(dir2 + 'frame_0007700000.pt', device=device, ucb=True),
        ])
        tournament.add_agent('Candidate-Version5', ensemble)

        dir1 = 'models/final-BBLN-crps-explore-adv-RERUN_20230809_11:10/'
        dir2 = 'models/final-BBLN-crps-explore-adv-RERUN_20230809_11:15/'
        agents = [
            NNAgent.load_model(dir1 + 'frame_0001400000.pt', device=device),
            NNAgent.load_model(dir1 + 'frame_0001500000.pt', device=device),
            NNAgent.load_model(dir2 + 'frame_0001400000.pt', device=device),
            NNAgent.load_model(dir2 + 'frame_0001500000.pt', device=device),
        ]
        ensemble = MajorityVoteAgent(agents)
        tournament.add_agent('RERUN-Live-V3', ensemble)

        dir1 = 'models/final-BBLN-crps-explore-adv-RERUN_20230809_11:10/'
        dir2 = 'models/final-BBLN-crps-explore-adv-RERUN_20230809_11:15/'
        agents = [
            NNAgent.load_model(dir1 + 'frame_0003600000.pt', device=device),
            NNAgent.load_model(dir1 + 'frame_0003700000.pt', device=device),
            NNAgent.load_model(dir2 + 'frame_0003600000.pt', device=device),
            NNAgent.load_model(dir2 + 'frame_0003700000.pt', device=device),
        ]
        ensemble = MajorityVoteAgent(agents)
        tournament.add_agent('RERUN-Candidate-V4', ensemble)

    tournament.add_agent('lilith_weak', NNAgent.load_model('baselines/lilith_weak.pt', device=device))
    tournament.add_agent('baseline1_LN', NNAgent.load_model('models/final-baseline1-LN_20230809_03:19/frame_0006000000.pt', device=device))
    tournament.add_agent('BBLN-crps-4.7M', NNAgent.load_model('models/final-BBLN-crps-explore_20230809_03:17/frame_0004700000.pt', device=device))
    tournament.add_agent('BBLN-crpsexp-SELF-100k', NNAgent.load_model('/mnt/qb/work2/goswami0/gkd021/code/rl-teamproject/models/final-BBLN-crps-explore-adv-RERUN_20230809_11:10/frame_0000100000.pt', device=device))
    
    tournament.add_agent('LSTM-5.3M', NNAgent.load_model('models/final-lstm-big_20230809_03:16/frame_0005300000.pt', device=device))
    agents = [
        NNAgent.load_model('models/final-lstm-big_20230809_03:16/frame_0005000000.pt', device=device),
        NNAgent.load_model('models/final-lstm-big_20230809_03:16/frame_0005100000.pt', device=device),
        NNAgent.load_model('models/final-lstm-big_20230809_03:16/frame_0005200000.pt', device=device),
        NNAgent.load_model('models/final-lstm-big_20230809_03:16/frame_0005300000.pt', device=device),
    ]
    tournament.add_agent('LSTM-ensemble', MajorityVoteAgent(agents))

    tournament.add_agent('LSTM-big-adv-phase2-6.3M', NNAgent.load_model('models/final-lstm-big-adv-phase2_20230809_07:30/frame_0006300000.pt', device=device))
    
    tournament.add_agent('BBLN-crps-5.6M', NNAgent.load_model('models/final-BBLN-crps_20230809_03:17/frame_0005600000.pt', device=device))
    tournament.add_agent('BBLN-crpsexp-5.6M', NNAgent.load_model('models/final-BBLN-crps-explore_20230809_03:17/frame_0005600000.pt', device=device))
    
    agent = NNAgent.load_model('models/final-BBLN-crps_20230809_03:17/frame_0005600000.pt', device=device)
    agent.ucb = True
    tournament.add_agent('BBLN-crps-5.6M-UCB', agent)
    
    agent = NNAgent.load_model('models/final-BBLN-crps_20230809_03:17/frame_0005600000.pt', device=device)
    agent.greedy = True
    tournament.add_agent('BBLN-crps-5.6M-greedy', agent)

    tournament.add_agent('BBLN-crps-6.2M', NNAgent.load_model('models/final-BBLN-crps_20230809_03:17/frame_0006200000.pt', device=device))
    tournament.add_agent('BBLN-crpsexp-6.2M', NNAgent.load_model('models/final-BBLN-crps-explore_20230809_03:17/frame_0006200000.pt', device=device))

    if True:
        dir1 = 'models/final-BBLN-crps_20230809_03:17/'
        ensemble = MajorityVoteAgent([
            NNAgent.load_model(dir1 + 'frame_0005900000.pt', device=device),
            NNAgent.load_model(dir1 + 'frame_0006000000.pt', device=device),
            NNAgent.load_model(dir1 + 'frame_0006100000.pt', device=device),
            NNAgent.load_model(dir1 + 'frame_0006200000.pt', device=device),
        ])
        tournament.add_agent('CANDIDATE-1-prime', ensemble)

        dir2 = 'models/final-BBLN-crps-explore_20230809_03:17/'
        ensemble = MajorityVoteAgent([
            NNAgent.load_model(dir2 + 'frame_0005900000.pt', device=device),
            NNAgent.load_model(dir2 + 'frame_0006000000.pt', device=device),
            NNAgent.load_model(dir2 + 'frame_0006100000.pt', device=device),
            NNAgent.load_model(dir2 + 'frame_0006200000.pt', device=device),
        ])
        tournament.add_agent('CANDIDATE-2-prime', ensemble)


    print(f'Playing {args.games} games...')
    try:
        for _ in tournament.random_plays(n_plays=args.games, verbose=not args.quiet):
            pass 
    except KeyboardInterrupt:
        pass
    elos = tournament.leaderboard.mean_elos()
    elos_sorted = list(sorted(elos.items(), key=lambda x: x[1], reverse=True))
    pprint.pprint(elos_sorted, indent=4)

if __name__ == '__main__':
    main()