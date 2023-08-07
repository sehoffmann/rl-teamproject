import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from collections import deque
import time

class MetricList:

    def __init__(self):
        self.metrics = {}

    def add(self, key, value):
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)

    def clear(self, key=None):
        if key is None:
            self.metrics = {}
        else:
            del self.metrics[key] 

    def __getitem__(self, key):
        return self.metrics.get(key, [])
    
    def __contains__(self, key):
        return key in self.metrics

    def __iter__(self):
        return iter(self.metrics)
    


class Tracker:
    def __init__(self, wandb=True, tracking_frequency=5_000):
        self.wandb = wandb
        self.tracking_frequency = tracking_frequency

        self.num_frames = 0
        self.num_updates = 0
        self.num_games = 0

        self.interval_metrics = MetricList()
        self.interval_start = None
        self.interval_frame_idx = 0
        self.episode_frame_idx = 0
        self.episode_cum_reward = 0
        self.winner_stats = []
        self.reset()
    
    def reset(self):
        self.interval_start = time.time()
        self.interval_frame_idx = self.num_frames
        self.interval_metrics.clear()
        self.winner_stats = []

    def _finalize_interval(self):
        # FPS
        time_elapsed = time.time() - self.interval_start
        frames_elapsed = self.num_frames - self.interval_frame_idx
        fps = frames_elapsed / time_elapsed

        # Winner stats
        winner_stats = np.array(self.winner_stats)
        draw_rate = np.mean(winner_stats == 0) * 100
        win_rate = np.mean(winner_stats == 1) * 100
        loss_rate = np.mean(winner_stats == 2) * 100

        loss = np.mean(self.interval_metrics['loss'])
        reward = np.mean(self.interval_metrics['episode_reward'])

        print(f'Fr. {self.num_frames: >7} | Game {self.num_games: >5} | Loss {loss:.2f} | Rew. {reward:.3f} | WR {win_rate:.1f}% | {fps:.0f} FPS')

        self.reset()

    def add_frame(self, reward):
        self.num_frames += 1
        self.episode_cum_reward += reward
        if self.num_frames % self.tracking_frequency == 0:
            self._finalize_interval()

    def add_update(self, loss):
        self.num_updates += 1
        self.interval_metrics.add('loss', loss)

    def add_game(self, info):
        self.num_games += 1
        self.interval_metrics.add('episode_length', self.num_frames - self.episode_frame_idx)
        self.interval_metrics.add('episode_reward', self.episode_cum_reward)

        self.episode_cum_reward = 0
        self.episode_frame_idx = self.num_frames
        self.winner_stats.append(info['winner'])

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