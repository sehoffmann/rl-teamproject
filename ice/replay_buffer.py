from collections import deque
from typing import Dict, List, Tuple
import numpy as np
import random
import torch
from segment_tree import *
from decay import BetaDecay


class ReplayBuffer:
    """A simple numpy replay buffer.
    Parameters
    ---------
    obs_dim: list[int]
        Observation shape
    size: int
        # maximum number of elements in buffer
    batch_size: int
        batch_size
    n_step: int
        number of step used for N-step learning
    gamma: float
        gamma value
    """

    def __init__(
            self,
            obs_dim: List[int],
            size: int = 1024,
            batch_size: int = 32,
            n_step: int = 1,
            gamma: float = 0.99
    ):
        self.obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
            self,
            obs: np.ndarray,
            act: int,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool] or None:
        """Store a new experience in the buffer"""
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return

        # make a n-step transition
        # take the n-reward, n-observation and n-done
        rew, next_obs, done = self._get_n_step()
        # take the 1-observation and 1-action
        obs, act = self._get_first_step()

        # store the transition
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        return transition

    def sample_batch_torch(self, num_frames=1, device=None):
        batch = self.sample_batch(num_frames)
        batch_torch = {
            'obs': torch.FloatTensor(batch['obs']).to(device, non_blocking=True),
            'next_obs': torch.FloatTensor(batch['next_obs']).to(device, non_blocking=True),
            'acts': torch.LongTensor(batch['acts']).to(device, non_blocking=True),
            'rews': torch.FloatTensor(batch['rews']).to(device, non_blocking=True),
            'done': torch.FloatTensor(batch['done']).to(device, non_blocking=True),
            'weights': torch.FloatTensor(batch['weights']).to(device, non_blocking=True),
            'indices': batch['indices']
        }
        return batch_torch

    def sample_batch(self, num_frames=1):
        """Sample a batch from the buffer"""
        assert len(self) >= self.batch_size
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            weights=np.ones(self.batch_size, dtype=np.float32),
            indices=idxs,
        )

    def sample_batch_from_idxs(self, idxs: np.ndarray):
        """Sample a batch given some fixed idxs"""
        # for N-step Learning
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def _get_n_step(self) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]
            # update the reward
            rew = r + self.gamma * rew * (1 - d)
            # if done == 1: next_obs is the first observation where done == 1
            # if done == 0: next_obs is the n-observation
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def _get_first_step(self) -> Tuple[np.int64, np.ndarray]:
        """Return first step obs and act."""
        # info of the first transition
        obs, act = self.n_step_buffer[0][:2]

        return obs, act

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.
    Attributes:
        max_priority: float
            max priority
        tree_ptr: int
            next index of tree
        alpha: float
            alpha parameter for prioritized replay buffer
        sum_tree: SumSegmentTree
            sum tree for prior
        min_tree: MinSegmentTree
            min tree for min prior to get max weight
    """

    def __init__(
            self,
            obs_dim: List[int],
            size: int = 1024,
            batch_size: int = 32,
            alpha: float = 0.6,
            n_step: int = 1,
            gamma: float = 0.99,
            beta_start = 0.4,
            beta_max = 1.0,
            beta_frames = 5_000_000,
            epsilon = 1e-6,
    ):
        """Initialization."""
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(
            obs_dim, size, batch_size, n_step, gamma
        )
        self.max_priority = 1.0
        self.tree_ptr = 0
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta_decay = BetaDecay(beta_start, beta_max, beta_frames)

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
            self,
            obs: np.ndarray,
            act: int,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
    ) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
        """Store an experience and its priority."""
        transition = super().store(obs, act, rew, next_obs, done)

        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size

        return transition

    def sample_batch(self, num_frames=1):
        return self.sample_batch_beta(self.beta_decay(num_frames))

    def sample_batch_beta(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0

        # samples transitions indices
        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        # importance sampling weights
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,  # need this for priority updating
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        priorities += self.epsilon  # avoid zero priority
        self.max_priority = max(self.max_priority, np.max(priorities))
        alpha_prios = priorities ** self.alpha

        for idx, alpha_prio in zip(indices, alpha_prios):
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = alpha_prio
            self.min_tree[idx] = alpha_prio

    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self))
        segment = p_total / self.batch_size

        # perform a random sample in each segment
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upper_bound = random.uniform(a, b)
            idx = self.sum_tree.find_prefixsum_idx(upper_bound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight


import numpy as np

class FrameStack:
    """A container storing a limited amount of stacked observations,
    it is possible to add new frames and remove the oldest ones like a deque"""
    def __init__(self, dimension=1, mode="array"):
        self.frames = None  # [H, W, C] or [W]
        self.mode = mode  # 'array' or 'pixels'
        self.dimension = dimension  # how many frames to stack together
        self.curr_frames = 0
        self.w = None
        self.h = None
        self.c = None

    def unstack(self):
        """Unstack the oldest frame"""
        assert self.w is not None and self.curr_frames > 0
        if self.mode == "array":
            self.frames = self.frames[self.w:]
        if self.mode == "pixels":
            self.frames = self.frames[self.h:]
        self.curr_frames -= 1

    def stack(self, img, repeat=1):
        """Stack a frame (img) at the end repeated n times"""
        assert 0 <= repeat
        for _ in range(repeat):
            if self.mode == "array":
                if self.curr_frames == 0:
                    self.w = img.shape[0]
                    self.frames = img
                else:
                    self.frames = np.concatenate((self.frames, img), axis=-1)
            if self.mode == "pixels":
                if self.curr_frames == 0:
                    self.w = img.shape[1]
                    self.h = img.shape[0]
                    self.c = img.shape[2]
                    self.frames = img
                else:
                    self.frames = np.concatenate((self.frames, img), axis=0)
            self.curr_frames += 1
            if self.curr_frames > self.dimension:
                self.unstack()
        return self.frames

    def clear(self):
        """Remove all frames"""
        while self.curr_frames > 0:
            self.unstack()

    def full(self):
        """Return True if the stack is full"""
        return self.curr_frames == self.dimension

    def __len__(self):
        return self.curr_frames

    def get_frames(self):
        return self.frames

from collections import deque

class FrameStacker2:
    def __init__(self, num_frames=1):
        self.num_frames = num_frames  # how many frames to stack together
        self.buffer = deque(maxlen=num_frames)
        self._stacked = None

    @property
    def stacked_frames(self):
        if self._stacked is not None:
            return self._stacked
        
        assert len(self.buffer) > 0
        if self.num_frames == 1:
            self._stacked = self.buffer[0]
        else:
            buffer = self.buffer
            if len(buffer) < self.num_frames:
                n_repeats = self.num_frames - self.num_frames
                buffer = [buffer[0]] * n_repeats + list(self.buffer) # repeat the first element
            self._stacked_frames = np.concatenate(buffer, axis=-1)
        return self._stacked

    def clear(self):
        self._stacked = None
        self.buffer.clear()

    def append(self, frame):
        self._stacked = None
        self.buffer.append(frame)

    def append_and_stack(self, frame):
        self.append(frame)
        return self.stacked_frames

    def is_full(self):
        return len(self) == self.num_frames

    def __len__(self):
        return len(self.buffer)