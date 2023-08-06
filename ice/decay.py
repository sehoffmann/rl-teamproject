class EpsilonDecay:

    def __init__(self, start_eps=1.0, min_eps=0.1, num_frames=1_000_000, constant_eps=None):
        self.start_eps = start_eps
        self.min_eps = min_eps
        self.num_frames = num_frames
        self.constant_eps = constant_eps

    def __call__(self, frames):
        if self.constant_eps is not None:
            return self.constant_eps
        else:
            decay_factor = 1 / self.num_frames
            eps = self.start_eps - (self.start_eps - self.min_eps) * decay_factor * frames
            return max(self.min_eps, eps)


class BetaDecay:

    def __init__(self, start_beta=0.4, max_beta=1.0, num_frames=5_000_000, constant_beta=None):
        self.start_beta = start_beta
        self.max_beta = max_beta
        self.num_frames = num_frames
        self.constant_beta = constant_beta

    def __call__(self, frames):
        if self.constant_beta is not None:
            return self.constant_beta
        else:
            decay_factor = 1 / self.num_frames
            beta = self.start_beta + (self.max_beta - self.start_beta) * decay_factor * frames
            return min(self.max_beta, beta)