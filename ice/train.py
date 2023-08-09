import argparse
import wandb
import torch
import datetime
import os
from pprint import pprint
import json
from pathlib import Path

from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from environments import IcyHockey
from models import Lilith, Baseline1, LSTM
from decay import EpsilonDecay
from dqn import NNAgent, DqnAgent, DqnTrainer, TRAINING_SCHEDULES

MODELS = ['lilith', 'lilith_big', 'baseline1', 'baseline1_layernorm', 'baseline1_ln_big', 'LSTM-small', 'LSTM-big']

def create_model(config, num_actions, obs_shape):
    if config['crps']:
        num_actions *= 2 # predict both mean and std

    cp_path = config['checkpoint']
    if cp_path:
        print(f'Loading model from {cp_path}')
        return torch.load(cp_path)
    elif config['model'] == 'lilith':
        model = Lilith(
            obs_shape[1]*obs_shape[0], 
            num_actions, 
            hidden_size=256, 
            dueling=config['dueling'],
        )
        return model
    elif config['model'] == 'lilith_big':
        model = Lilith(
            obs_shape[1]*obs_shape[0], 
            num_actions, 
            hidden_size=512, 
            dueling=config['dueling'],
        )
        return model
    elif config['model'] == 'baseline1':
        model = Baseline1(
            obs_shape[1]*obs_shape[0], 
            num_actions, 
            hidden_size=512, 
            n_hidden_layers=3,
            layer_norm=False
        )
        return model
    elif config['model'] == 'baseline1_layernorm':
        model = Baseline1(
            obs_shape[1]*obs_shape[0], 
            num_actions, 
            hidden_size=512, 
            n_hidden_layers=3,
            layer_norm=True
        )
        return model
    elif config['model'] == 'baseline1_ln_big':
        model = Baseline1(
            obs_shape[1]*obs_shape[0], 
            num_actions, 
            hidden_size=512, 
            n_hidden_layers=5,
            n_hidden_heads=2,
            layer_norm=True
        )
        return model
    elif config['model'] == 'LSTM-small':
        model = LSTM(
            obs_shape[1], 
            num_actions, 
            hidden_size=128,
            num_head_layers=1,
            dueling=config['dueling'],
        )
        return model
    elif config['model'] == 'LSTM-big':
        model = LSTM(
            obs_shape[1], 
            num_actions, 
            hidden_size=256,
            num_feature_layers=4,
            dueling=config['dueling'],
        )
        return model

def train(config, model_dir, device):
    ## ENV
    env = IcyHockey(reward_shaping=config['reward_shaping'])

    # Replay Buffer
    obs_shape = [config['frame_stacks'], env.observation_space.shape[0]]
    if not config['priority_rp']:
        replay_buffer = ReplayBuffer(obs_shape, config['buffer_size'], config['batch_size'], n_step=config['nsteps'])
    else:
        replay_buffer = PrioritizedReplayBuffer(
            obs_shape, 
            config['buffer_size'], 
            config['batch_size'], 
            n_step = config['nsteps'],
            gamma = config['gamma'],
            beta_frames = config['beta_decay'],
        )

    # Model & DQN Agent
    model = create_model(config, env.action_space.n, obs_shape).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr']) #
    if config['cosine_annealing']:
        scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['frames'], 1e-6)
    else:
        scheduler = None

    if config['rampup'] > 0:
        lr_rampup = torch.torch.optim.lr_scheduler.LinearLR(optimizer, 1e-1, total_iters=config['rampup'])
        schedulers = [lr_rampup]
        if scheduler is not None:
            schedulers.append(scheduler)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedulers)

    epsilon_decay = EpsilonDecay(num_frames=config['eps_decay'])
    dqn_agent = DqnAgent(
        model, 
        optimizer, 
        env.action_space.n,
        device,
        frame_stacks=config['frame_stacks'],
        epsilon_decay=epsilon_decay, 
        gamma=config['gamma']**config['nsteps'],
        no_double=not config['double_q'],
        scheduler=scheduler,
        softactions=config['softactions'],
        crps=config['crps'],
        crps_explore=config['crps_explore'],
    )

    assert config['warmup_frames'] >= config['bootstrap_frames']

    # Trainer
    trainer = DqnTrainer(
        model_dir,
        env, 
        dqn_agent, 
        replay_buffer, 
        device,
        frame_stacks=config['frame_stacks'],
        update_frequency=config['update_frequency'],
        training_delay=config['warmup_frames'] - config['bootstrap_frames'],
        schedule=config['schedule'],
    )

    # Prepopulate using lilith-weak
    if config['bootstrap_frames'] > 0:
        lilith_weak = NNAgent.load_lilith_weak(device)
        trainer.prepopulate(lilith_weak, config['bootstrap_frames'])

    trainer.train(config['frames'])

    wandb.finish()


def create_model_dir(config):
    model_dir = Path(f'models') / f'{config["name"].replace(" ", "_")}_{datetime.datetime.now().strftime("%Y%m%d_%H:%M")}'
    os.makedirs(model_dir, exist_ok=True)
    print('Model dir:', model_dir.resolve())
    return model_dir


def init_wandb(config, args):
    wandb_mode = 'disabled' if args.no_wandb else 'online'
    wandb_name = None if config['name'] == 'test' else config['name']
    wandb.init(project='ice', name=wandb_name, mode=wandb_mode, reinit=True)
    wandb.config.update(config)

def make_config(args):
    config = {
        'frames': args.frames,
        'name': args.name,
        'checkpoint': args.checkpoint,
        'model': args.model,
        "schedule": args.schedule,
        'priority_rp': args.per,
        'double_q': args.double_q,
        'nsteps': args.nsteps,
        'dueling': not args.no_dueling,
        'frame_stacks': args.frame_stacks,
        'gamma': args.gamma,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'cosine_annealing': args.cosine_annealing,
        'update_frequency': args.update_frequency,
        'warmup_frames': args.warmup_frames,
        'buffer_size': args.buffer_size,
        'eps_decay': args.eps_decay,
        'beta_decay': args.beta_decay,
        'bootstrap_frames': args.bootstrap_frames,
        'rampup': args.rampup,
        'softactions': args.softactions,
        'crps': args.crps,
        'crps_explore': args.crps,
        'reward_shaping': not args.no_shaping,
    }
    return config

def run(config):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_dir = create_model_dir(config)
    print('Config:')
    pprint(config, indent=4)
    with open(model_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    train(config, model_dir, device)

    return model_dir

def main():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('-f', '--frames', type=int, default=4_000_000)
    parser.add_argument('-n', '--name', type=str, default='test')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--preset', type=str, default=0)

    # Other
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--warmup-frames', type=int, default=500_000)
    parser.add_argument('--buffer-size', type=int, default=500_000)
    parser.add_argument('--update-frequency', type=int, default=2)
    parser.add_argument('--eps-decay', type=int, default=1_000_000)
    parser.add_argument('--beta-decay', type=int, default=2_000_000)
    parser.add_argument('--gamma', type=float, default=0.99)

    # Method
    parser.add_argument('--model', choices=MODELS, type=str, default='lilith')
    parser.add_argument('--schedule', choices=TRAINING_SCHEDULES, type=str)
    parser.add_argument('--advanced-schedule', action='store_true')
    parser.add_argument('--per', action='store_true')
    parser.add_argument('--no-dueling', action='store_true')
    parser.add_argument('--nsteps', type=int, default=3)
    parser.add_argument('--double-q', action='store_true')
    parser.add_argument('--frame-stacks', type=int, default=1)
    parser.add_argument('--cosine-annealing', action='store_true')
    parser.add_argument('--rampup', type=int, default=0)
    parser.add_argument('--bootstrap-frames', type=int, default=0)
    parser.add_argument('--softactions', action='store_true')
    parser.add_argument('--crps', action='store_true')
    parser.add_argument('--crps-explore', action='store_true')
    parser.add_argument('--no-shaping', action='store_true')
 
    args = parser.parse_args()

    ## create config for phase 1 and phase 2
    config = make_config(args)
    
    if args.advanced_schedule:
        phase1_config = config.copy()
        phase1_config['eps_decay'] = 1_000_000
        phase1_config['frames'] = 2_000_000
        # phase1_config['frames'] = 100_000
        phase1_config['bootstrap_frames'] = 300_000
        phase1_config['cosine_annealing'] = True
        phase1_config['nsteps'] = 4
        phase1_config['name'] += '-phase1'
        phase1_config['schedule'] = 'adv1'
        
        assert phase1_config['frames'] % 100_000 == 0, "frames must be divisible by checkpoint_frequency to support advanced schedule"
        init_wandb(phase1_config, args)
        prev_modeldir = run(phase1_config)
        
        checkpoint = prev_modeldir / f"frame_{phase1_config['frames']:010d}.pt"

        phase2_config = config.copy()
        phase2_config['eps_decay'] = 1
        phase2_config['frames'] = 15_000_000
        # phase2_config['frames'] = 50_000
        phase2_config['name'] += '-phase2'
        phase2_config['schedule'] = 'basic'
        phase2_config['rampup'] = 1_000_000
        phase2_config['nsteps'] = 4
        phase2_config['cosine_annealing'] = True
        phase2_config['reward_shaping'] = False
        phase2_config['checkpoint'] = str(checkpoint)
        init_wandb(phase2_config, args)
        run(phase2_config)
    else:
        init_wandb(config, args)
        run(config)



if __name__ == '__main__':
    main()