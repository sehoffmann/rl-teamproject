import argparse
import wandb
import torch


from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from environments import DiscreteHockey_BasicOpponent
from models import DenseNet
from decay import EpsilonDecay
from dqn import DqnAgent, DqnTrainer

def create_model(args, num_actions, obs_shape):
    
    if args.checkpoint:
        return torch.load(args.checkpoint)
    else:
        model = DenseNet(
            obs_shape[0], 
            num_actions, 
            hidden_size=256, 
            no_dueling=args.no_dueling
        )
        return model

def train(args):
    wandb_mode = 'disabled' if args.no_wandb else 'online'
    wandb.init(project='ice', mode=wandb_mode)
    wandb.config.update(args)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    warmup_frames = 50_000

    frame_stacks = args.frame_stacks
    buffer_size = 500_000
    batch_size = 256
    lr = 1e-4
    gamma = 0.99
    eps_decay_frames = 1_000_000
    beta_decay_frames = 3_000_000
    update_frequency = 2

    if args.no_nstep:
        n_step = 1
    else:
        n_step = 4

    # ENV
    env = DiscreteHockey_BasicOpponent()
    
    # Replay Buffer
    obs_shape = [env.observation_space.shape[0] * frame_stacks]
    if args.no_per:
        replay_buffer = ReplayBuffer(obs_shape, buffer_size, batch_size)
    else:
        replay_buffer = PrioritizedReplayBuffer(
            obs_shape, 
            buffer_size, 
            batch_size, 
            n_step = n_step,
            gamma = gamma,
            beta_frames = beta_decay_frames
        )

    # Model & DQN Agent
    model = create_model(args, env.action_space.n, obs_shape).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epsilon_decay = EpsilonDecay(num_frames=eps_decay_frames)
    dqn_agent = DqnAgent(
        model, 
        optimizer, 
        env.action_space.n,
        device,
        epsilon_decay=epsilon_decay, 
        gamma=gamma**n_step,
        no_double=args.no_double,
    )

    # Trainer
    trainer = DqnTrainer(
        env, 
        dqn_agent, 
        replay_buffer, 
        device,
        frame_stacks=frame_stacks,
        update_frequency=update_frequency,
        training_delay=warmup_frames,
    )

    trainer.train(2_000_000)


def main():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--checkpoint', type=str)

    # Method
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--no-per', action='store_true')
    parser.add_argument('--no-dueling', action='store_true')
    parser.add_argument('--no-nstep', action='store_true')
    parser.add_argument('--no-double', action='store_true')
    parser.add_argument('--frame-stacks', type=int, default=1)
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()