import matplotlib.pyplot as plt
import pandas as pd
import wandb

class Tracker:
    def __init__(self, wandb=True):
        self.values = {}
        self.wandb = wandb
        self.num_frames = 0
        self.num_steps = 0
        self.num_games = 0

    def finish_frame(self):
        self.num_frames += 1

    def finish_step(self):
        self.num_steps += 1

    def finish_game(self, winner, score):
        self.num_games += 1

    def log(self, key, value):
        if self.wandb:
            wandb.log({key: value}, commit=False)
        
        if key not in self.values:
            self.values[key] = []
        self.values[key].append((step, value))

    def save_csv(self, path):
        df = pd.DataFrame()
        for key in self.values:
            df[key] = pd.Series([v[1] for v in self.values[key]], [v[0] for v in self.values[key]])
        df.to_csv(path)

    def plot(self, keys, title=None, smoothing=0.0):
        fig = plt.figure(figsize=(12,6))
        for key in keys:
            x = [v[0] for v in self.values[key]]
            vals = [v[1] for v in self.values[key]]
            if smoothing > 0:
                vals = pd.Series(vals).ewm(alpha=1-smoothing).mean()
            plt.plot(x, vals, label=key)
        plt.legend()
        plt.title(title)
        return fig