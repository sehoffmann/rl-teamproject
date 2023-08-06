import numpy as np

from dqn import DQNAgent
from discretization import DiscreteHockey_BasicOpponent

# Select an environment
env_name = ["CartPole-v0", "MsPacman-v0"]

# PARAMETERS---------------------------------------------------
num_frames = 4_000_000  # number of training frames
memory_size = 5_000_00  # replay memory size
batch_size = 128  # batch size
lr = 1e-4 #
target_update = 1000  # update target network frequency
frame_interval = 10000  # refresh plot frequency 
plot = True  # plot score and loss during training
model_name = 'test'  # model name, don't need to change it
training_delay = num_frames // 50  # number of frames before start training
trials = 100  # number of evaluation episodes
frames_stack = 1  # number of consecutive frames to take as input
train = True  # train a new model
test = False  # evaluate the new model if train==True,
             # otherwise try to load an old model that has been trained for num_frames frames
             # and if present use it to perform evaluation
# ---------------------------------------------------------------

preprocess_function = None
env = DiscreteHockey_BasicOpponent()

agent = DQNAgent(env, memory_size, batch_size, target_update,
                 plot=plot,
                 lr=lr,
                 frame_interval=frame_interval,
                 frame_preprocess=preprocess_function,
                 n_frames_stack=frames_stack,
                 model_name=model_name,
                 training_delay=training_delay,
                 no_categorical=True,
                 no_noise=True,
                 )

"""agent = DQNAgent(env, memory_size, batch_size, target_update,
                         no_dueling=True, no_categorical=True, no_double=True,
                         no_n_step=True, no_noise=True, no_priority=True,
                         plot=plot, frame_interval=frame_interval)"""
if train:
    score, loss = agent.train(num_frames)
    agent.save()

if test:
    best_score = -np.inf
    best_frames = []
    agent.load()
    tot_score = 0
    frames = []
    for i in range(trials):
        score, frames = agent.test(get_frames=True)
        tot_score += score
        print("Score: ", score)
        if score > best_score:
            best_score = score
            best_frames = frames
    print("Average score:", tot_score / trials)
    print("Best score:", best_score)

    env.close()