import argparse
import wandb
import torch
import datetime
import os
from pprint import pprint
import json
from pathlib import Path

from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from environments import DiscreteHockey_BasicOpponent
from models import DenseNet
from decay import EpsilonDecay
from dqn import DqnAgent, DqnTrainer

def create_model(config, num_actions, obs_shape):
    
    if config['checkpoint']:
        return torch.load(config['checkpoint'])
    else:
        model = DenseNet(
            obs_shape[0], 
            num_actions, 
            hidden_size=256, 
            no_dueling=not config['dueling'],
        )
        return model

def train(config, model_dir, device):
    # ENV
    env = DiscreteHockey_BasicOpponent()
    
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['frames'])
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
        training_delay=config['warmup_frames'],
    )

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
        'priority_rp': args.per,
        'double_q': args.double_q,
        'nsteps': args.nsteps,
        'dueling': not args.no_dueling,
        'frame_stacks': args.frame_stacks,
        'gamma': 0.99,
        'lr': 1e-4,
        'batch_size': 256,
        'cosine_annealing': args.cosine_annealing,
        'update_frequency': 2,
        'warmup_frames': 200_000,
        'buffer_size': 500_000,
        'eps_decay': 1_000_000,
        'beta_decay': 2_000_000,
    }
    return config

def main():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('-f', '--frames', type=int, default=4_000_000)
    parser.add_argument('-n', '--name', type=str, default='test')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--no-wandb', action='store_true')

    # Method
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