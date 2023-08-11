import argparse
import pprint
import torch
from elo_system import HockeyTournamentEvaluation
from dqn import NNAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints', nargs='*', default=[])
    parser.add_argument('-n', '--games', type=int, default=5000)
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('--no-basics', action='store_true')
    parser.add_argument('--ab', type=int, default=1)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tournament = HockeyTournamentEvaluation(add_basics=not args.no_basics, default_elos=False)
    tournament.add_agent('lilith-weak', NNAgent.load_lilith_weak(device=device))
    
    for cp in args.checkpoints:
        tournament.add_agent(cp, NNAgent.load_model(cp, device=device))
    
    if args.ab == 1:
        ### Ablation 1
        ablation = {
            "ab1-lilith-strong-bt150k": "models/ab1-lilith-strong-bt150k_20230808_17:19/frame_0005000000.pt",
            "ab1-lilith-strong-bt150k-nosoft": "models/ab1-lilith-strong-bt150k-nosoft_20230808_17:19/frame_0005000000.pt",
            "ab1-lilith-strong-bt150k-rampup": "models/ab1-lilith-strong-bt150k-rampup_20230808_17:19/frame_0005000000.pt",
            "ab1-lilith-strong-bt150k-rampup-nosoft": "models/ab1-lilith-strong-bt150k-rampup-nosoft_20230808_17:19/frame_0005000000.pt",
            "ab1-lilith-strong-bt300k": "models/ab1-lilith-strong-bt300k_20230808_17:19/frame_0005000000.pt",
            "ab1-lilith-strong-bt300k-nosoft": "models/ab1-lilith-strong-bt300k-nosoft_20230808_17:19/frame_0005000000.pt",
            "ab1-lilith-strong-bt300k-rampup": "models/ab1-lilith-strong-bt300k-rampup_20230808_17:19/frame_0005000000.pt",
            "ab1-lilith-strong-bt300k-rampup-nosoft": "models/ab1-lilith-strong-bt300k-rampup-nosoft_20230808_17:19/frame_0005000000.pt",
            "ab1-lilith-strong-bt300k-rampup": "models/ab1-lilith-strong-bt300k-rampup_20230808_17:19/frame_0005000000.pt",
            "ab1-lilith-strong-plain-nosoft": "models/ab1-lilith-strong-plain-nosoft_20230808_17:19/frame_0005000000.pt", 
            "ab1-lilith-strong-plain": "models/ab1-lilith-strong-plain_20230808_17:19/frame_0005000000.pt",
        }

    elif args.ab == 2:
        ablation = {
            "ab2-nsteps1": "models/ab2-nsteps1_20230808_17:20/frame_0005000000.pt",  
            "ab2-nsteps2": "models/ab2-nsteps2_20230808_17:21/frame_0005000000.pt",  
            "ab2-nsteps3": "models/ab2-nsteps3_20230808_17:21/frame_0005000000.pt",
            "ab2-nsteps4": "models/ab2-nsteps4_20230808_17:21/frame_0005000000.pt", 
            "ab2-nsteps5": "models/ab2-nsteps5_20230808_17:21/frame_0005000000.pt",
        }
    
    elif args.ab == 3:
        ablation = {
            "ab2-nsteps4": "models/ab2-nsteps4_20230808_17:21/frame_0005000000.pt",
            "ab3-DoubleQ": "models/ab3-DoubleQ_20230810_00:20/frame_0005000000.pt" ,
            "ab3-DoubleQ-Dueling": "models/ab3-DoubleQ-Dueling_20230810_00:20/frame_0005000000.pt" ,
            "ab3-DoubleQ-Dueling-PER": "models/ab3-DoubleQ-Dueling-PER_20230810_00:20/frame_0005000000.pt" ,
            "ab3-nothing": "models/ab3-nothing_20230810_00:20/frame_0005000000.pt" ,
            "ab3-PER-Dueling": "models/ab3-PER-Dueling_20230810_00:20/frame_0005000000.pt",
            "ab3-PER": "models/ab3-PER_20230810_00:18/frame_0005000000.pt",
        }
    
    elif args.ab == 4:
        ablation = {
            "ab2-nsteps4": "models/ab2-nsteps4_20230808_17:21/frame_0005000000.pt", # <- this has no bootstrapping !
            "ab4-baseline1":"models/ab4-baseline1_20230808_22:32/frame_0005000000.pt",
            "ab4-baseline1_layernorm":"models/ab4-baseline1_layernorm_20230808_22:32/frame_0005000000.pt",
            "ab4-baseline1_ln_big":"models/ab4-baseline1_ln_big_20230808_22:32/frame_0005000000.pt",
        }

    elif args.ab == 5:
        ablation = {
            "ab4-baseline1_ln_big": "models/ab4-baseline1_ln_big_20230808_22:32/frame_0005000000.pt",
            "ab5-baseline1_layernorm_stack": "models/ab5-baseline1_layernorm_stack_20230808_22:45/frame_0005000000.pt",
            "ab5-baseline1_ln_big_stack": "models/ab5-baseline1_ln_big_stack_20230808_22:46/frame_0005000000.pt", 
            "ab5-lilith_stack": "models/ab5-lilith_stack_20230808_22:45/frame_0005000000.pt", 
            "ab5-LSTM-big": "models/ab5-LSTM-big_20230808_22:46/frame_0005000000.pt", 
            "ab5-LSTM-small": "models/ab5-LSTM-small_20230808_22:45/frame_0005000000.pt",
        }

    elif args.ab == 6:
        ablation = {
            "ab4-baseline1_ln": "models/ab4-baseline1_layernorm_20230808_22:32/frame_0005000000.pt",
            "ab4-baseline1_ln_big": "models/ab4-baseline1_ln_big_20230808_22:32/frame_0005000000.pt",
            "ab1-lilith-strong": "models/ab1-lilith-strong-bt300k-nosoft_20230808_17:19/frame_0005000000.pt",
            "ab6-base1lnbig": "models/ab6-base1lnbig_20230809_02:37/frame_0005000000.pt", 
            "ab6-base1lnbig-exp":"models/ab6-base1lnbig-explore_20230809_02:37/frame_0005000000.pt", 
            "ab6-lilith":"models/ab6-lilith_20230809_02:36/frame_0005000000.pt", 
            "ab6-lilith-exp":"models/ab6-lilith-explore_20230809_02:36/frame_0005000000.pt", 
        }

    for name, cp in ablation.items():
        tournament.add_agent(name, NNAgent.load_model(cp, device=device))    

    print(f'Playing {args.games} games...')
    i = 1
    for _ in tournament.random_plays(n_plays=args.games, verbose=not args.quiet):
        if i % 500 == 0:
            print(f'Played {i} games')
            elos = tournament.leaderboard.mean_elos()
            elos_sorted = list(sorted(elos.items(), key=lambda x: x[1], reverse=True))
            pprint.pprint(elos_sorted, indent=4)
            print()
            tournament.leaderboard.save(f'evaluation/ab{args.ab}.json')  
        i += 1

if __name__ == '__main__':
    main()