import argparse
import wandb
import torch


from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from environments import DiscreteHockey_BasicOpponent
from models import DenseNet
from decay import EpsilonDecay
from dqn import DqnAgent, DqnTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--no-per', action='store_true')
    args = parser.parse_args()

    wandb_mode = 'disabled' if args.no_wandb else 'online'
    wandb.init(project='ice', mode=wandb_mode)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    warmup_frames = 50_000

    frame_stacks = 1
    buffer_size = 500_000
    batch_size = 128
    lr = 1e-4
    n_step = 3
    gamma = 0.99
    eps_decay_frames = 1_000_000
    beta_decay_frames = 3_000_000
    update_frequency = 2

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
    num_actions = env.action_space.n
    model = DenseNet(
        obs_shape[0], 
        num_actions, 
        hidden_size=256, 
        no_dueling=False
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epsilon_decay = EpsilonDecay(num_frames=eps_decay_frames)
    dqn_agent = DqnAgent(
        model, 
        optimizer, 
        num_actions,
        device,
        epsilon_decay=epsilon_decay, 
        gamma=gamma**n_step,
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


if __name__ == '__main__':
    main()