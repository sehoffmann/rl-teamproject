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
from models import Lilith
from decay import EpsilonDecay
from dqn import NNAgent, DqnAgent, DqnTrainer, TRAINING_SCHEDULES

MODELS = ['lilith']

def create_model(config, num_actions, obs_shape):
    cp_path = config['checkpoint']
    if cp_path:
        print(f'Loading model from {cp_path}')
        return torch.load(cp_path)
    elif config['model'] == 'lilith':
        model = Lilith(
            obs_shape[0], 
            num_actions, 
            hidden_size=256, 
            dueling=config['dueling'],
        )
        return model

def train(config, model_dir, device):
    # ENV
    env = IcyHockey()
    
    # Replay Buffer
    obs_shape = [env.observation_space.shape[0] * config['frame_stacks']]
    if not config['priority_rp']:
        replay_buffer = ReplayBuffer(obs_shape, config['buffer_size'], config['batch_size'])
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
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    if config['cosine_annealing']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['frames'], 1e-6)
    else:
        scheduler = None
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
    )

    # Trainer
    trainer = DqnTrainer(
        model_dir,
        env, 
        dqn_agent, 
        replay_buffer, 
        device,
        frame_stacks=config['frame_stacks'],
        update_frequency=config['update_frequency'],
        training_delay=0 if config['lilith_bootstrap'] else config['warmup_frames'],
        schedule=config['schedule'],
    )

    # Prepopulate using lilith-weak
    if config['lilith_bootstrap']:
        lilith_weak = NNAgent.load_lilith_weak(device)
        trainer.prepopulate(lilith_weak, config['warmup_frames'])

    trainer.train(config['frames'])


def create_model_dir(config):
    model_dir = Path(f'models') / f'{config["name"].replace(" ", "_")}_{datetime.datetime.now().strftime("%Y%m%d_%H:%M")}'
    os.makedirs(model_dir, exist_ok=True)
    print('Model dir:', model_dir.resolve())
    return model_dir


def init_wandb(config, args):
    wandb_mode = 'disabled' if args.no_wandb else 'online'
    wandb_name = None if config['name'] == 'test' else args.name
    wandb.init(project='ice', name=wandb_name, mode=wandb_mode)
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
        'lilith_bootstrap': not args.no_lilith_bootstrap,
    }
    return config

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
    parser.add_argument('--no-lilith-bootstrap', action='store_true')

    # Method
    parser.add_argument('--model', choices=MODELS, type=str, default='lilith')
    parser.add_argument('--schedule', choices=TRAINING_SCHEDULES, type=str)
    parser.add_argument('--per', action='store_true')
    parser.add_argument('--no-dueling', action='store_true')
    parser.add_argument('--nsteps', type=int, default=3)
    parser.add_argument('--double-q', action='store_true')
    parser.add_argument('--frame-stacks', type=int, default=1)
    parser.add_argument('--cosine-annealing', action='store_true')

    args = parser.parse_args()
    config = make_config(args)

    init_wandb(config, args)

    model_dir = create_model_dir(config)
    print('Config:')
    pprint(config, indent=4)
    with open(model_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train(config, model_dir, device)



if __name__ == '__main__':
    main()