import os

import wandb

import laserhockey.laser_hockey_env as lh
from stable_baselines3 import PPO, SAC, DQN, TD3
from wandb.integration.sb3 import WandbCallback
from environments import LaserHockeyWithOpponent, LaserHockeyWithOpponentAndDiscreteActions

sweep_config = {
    "method": "bayes",
    "metric": {
        "goal": "maximize",
        "name": "ep_reward_mean",
    },
    "parameters": {
        "learning_rate": {"min": 1e-4, "max": 1e-2},
        "algorithm": {"values": ["SAC"]},
        "policy_type": {"values": ["MlpPolicy"]},
        "total_timesteps": {"values": [1e7]},
    }
    # "env_id": "1",
    # "ent_coef": 0.1,
}

MODEL_SAVE_FREQ = 1e5

os.environ['WANDB__SERVICE_WAIT'] = str(240)

sweep_id = wandb.sweep(sweep_config, project="sb3-hockey")


def benchmark_baseline(config=None):
    with wandb.init(project="sb3", config=config, sync_tensorboard=True, notes="Find the right algorithm.") as run:
        config = wandb.config
        
        agent2 = lh.BasicOpponent()
        # env = LaserHockeyWithOpponentAndDiscreteActions(opponent=agent2, mode=lh.LaserHockeyEnv.TRAIN_DEFENSE)

        env = LaserHockeyWithOpponent(opponent=agent2, mode=lh.LaserHockeyEnv.TRAIN_DEFENSE)

        if config.algorithm == "SAC":
            model = SAC(config.policy_type, 
                        env, 
                        verbose=0, 
                        seed=None, 
                        tensorboard_log=f"runs/{run.id}")
        elif config.algorithm == "TD3":
            model = TD3('MlpPolicy', 
                        env, verbose=1, 
                        seed=None, 
                        tensorboard_log=f"runs/{run.id}")
        elif config.algorithm == "PPO":
            model = PPO('MlpPolicy', 
                        env, 
                        verbose=1, 
                        seed=None, 
                        tensorboard_log=f"runs/{run.id}", 
                        ent_coef=config.ent_coef, 
                        learning_rate=config.learning_rate)
        elif config.algorithm == "DQN":
            env = LaserHockeyWithOpponentAndDiscreteActions(opponent=agent2, mode=lh.LaserHockeyEnv.TRAIN_DEFENSE)
            model = DQN('MlpPolicy',
                        env, 
                        verbose=1, 
                        seed=None, 
                        tensorboard_log=f"runs/{run.id}", 
                        learning_rate=config.learning_rate)
            
        # TODO: Not implemented
        if wandb.run.resumed:
            model = PPO.load(f"models/{wandb.run.id}/model", env, ent_coef=wandb.config.get("ent_coef"))

        #######
        ###### Training Curriculum
        ########

        MODE_DURATION = config.total_timesteps//3

        wandb_callback =WandbCallback(model_save_path=f"models/{run.id}",
                                      model_save_freq=MODEL_SAVE_FREQ,
                                      verbose=2)

        # 1. Defense, Defense, get'em get'em 
        env.mode = lh.LaserHockeyEnv.TRAIN_DEFENSE
        model.learn(total_timesteps=MODE_DURATION, callback=wandb_callback)
        
        # 2. Offense
        env.mode = lh.LaserHockeyEnv.TRAIN_SHOOTING
        model.learn(total_timesteps=MODE_DURATION, callback=wandb_callback)

        # 3. Game Play, esay mode
        env.mode = lh.LaserHockeyEnv.NORMAL
        model.learn(total_timesteps=MODE_DURATION, callback=wandb_callback)

        # 4. Refine against the hard opponent
        # env.mode = lh.LaserHockeyEnv.NORMAL
        # env.opponent = lh.BasicOpponent()

        run.finish()


# if __name__=="__main__":

    



#     run = wandb.init(
#         project="sb3",
#         config=config,
#         sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
#         # monitor_gym=True,  # auto-upload the videos of agents playing the game
#         # save_code=True,  # optional
#         notes="Find the right algorithm.",
#     )

wandb.agent(sweep_id, benchmark_baseline, count=3)