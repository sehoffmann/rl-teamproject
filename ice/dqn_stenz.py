from gymnasium import spaces
import numpy as np
import torch
from laserhockey.hockey_env import HockeyEnv

def get_stenz():
    env = HockeyEnv()
    stenz = DQNAgent(env.observation_space, env.action_space, eps=0.0)
    stenz.load_checkpoint('baselines/stenz.pth')
    return stenz

class DQNAgent(object):
    """
    Represents a DQNAgent that uses a Deep Q-Network to approximate the Q-values.
    """
    def __init__(self, observation_space, action_space, **userconfig):    
        # Spaces
        self._observation_space = observation_space
        self._action_space = spaces.Box(-1, 1, (4,))
        
        # Actions
        self._action_n = action_space.shape[0]
        self._total_action_dim = int(2**(self._action_n/2))
        
        # Configuration with default hyperparameters
        self._config = {
            "eps": 0.95,            # Initial epsilon for epsilon-greedy exploration
            "discount": 0.99,      # Discount factor for future rewards (gamma)
            "buffer_size": int(3e5), # Size of the experience replay buffer
            "batch_size": 64,     # Batch size for sampling from buffer
            "learning_rate": 0.00007, # Learning rate for neural network optimization
            "use_target_net": True  # Use a separate target network for stability
        }
        self._config.update(userconfig)        
        self._eps = self._config['eps']
        
        # Epsilon decay factors for exploration-exploitation balance
        self.eps_decay = 0.9995
        self.min_eps = 0.05
        
        # Initialize experience replay buffer with prioritized sampling
        self.buffer = PrioritizedMemory(max_size=self._config["buffer_size"])
                
        # Define primary Q-network and target Q-network
        self.Q = self._build_Q_network()
        self.Q_target = self._build_Q_network(learning_rate=0)
        
        # Ensure that the target network starts with the same weights as the primary network
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.train_iter = 0
            
    def _build_Q_network(self, learning_rate=None):
        """Construct the Q-network (Dueling DQN) for function approximation."""
        if learning_rate is None:
            learning_rate = self._config["learning_rate"]
        return QFunction(observation_dim=self._observation_space.shape[0], 
                         action_dim=self._total_action_dim,
                         learning_rate=learning_rate)

    def _update_target_net(self):
        """Soft update of the target network using weights of the primary network."""
        tau = 0.005
        for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def act(self, observation, eps=None):
        """Epsilon-greedy policy for action selection."""
        if eps is None:
            eps = self._eps
        if np.random.random() > eps:
            action = self.Q.greedyAction(observation)
        else:
            # Clip and round the sampled action values
            action = np.round(np.clip(self._action_space.sample(), -1, 1))
        return action
    
    def store_transition(self, transition):
        """Store a single transition into the replay buffer."""
        self.buffer.add_transition(transition)
    
    def train(self, iter_fit=8):
        """Training the agent using samples from the replay buffer."""
        if self.buffer.size > 30e3:
            losses = []
            self.train_iter += 1
            if self._config["use_target_net"]:
                self._update_target_net()

            for _ in range(iter_fit):
                data, weights = self.buffer.sample(batch=self._config['batch_size'])
                losses.append(self._optimize_model(data, weights))

            self._eps = max(self.min_eps, self._eps * self.eps_decay)  
        else:
            losses = [0]
        return losses

    def _optimize_model(self, data, weights):
        """Optimize the model using sampled data from the replay buffer."""
        s = np.stack(data[:,0])  # s_t
        a = np.stack(data[:,1])  # a_t
        rew = np.stack(data[:,2])[:, None]  # rew (batchsize,1)
        s_prime = np.stack(data[:,3])  # s_t+1
        done = np.stack(data[:,4])[:, None]  # done signal (batchsize,1)

        # Double DQN modification:
        # Find the best action using the primary Q network
        best_action = np.squeeze(np.argmax(self.Q.predict(s_prime), axis=-1, keepdims=True))

        # Get the Q-value of this action from the target network
        if self._config["use_target_net"]:
            v_prime = self.Q_target.Q_value(torch.from_numpy(s_prime).float(), torch.from_numpy(best_action))
            v_prime = v_prime.detach().numpy() # Convert tensor to numpy array
        else:
            v_prime = self.Q.Q_value(torch.from_numpy(s_prime).float(), torch.from_numpy(best_action))
            v_prime = v_prime.detach().numpy() # Convert tensor to numpy array

        gamma = self._config['discount']                                                
        td_target = rew + gamma * (1.0-done) * v_prime

         # optimize the lsq objective
        fit_loss = self.Q.fit(s, a, td_target)#*weights
        
        # update priorities
        with torch.no_grad():
            actions = np.apply_along_axis(lambda x: action_to_index(x), 1, a)
            td_errors = td_target - self.Q.Q_value(torch.from_numpy(s).float(), torch.from_numpy(actions)).numpy()
        self.buffer.update_priorities(td_errors)
        
        return fit_loss.mean()
    
    def save_checkpoint(self, path):
        checkpoint = {
            'model_state_dict': self.Q.state_dict()
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint['model_state_dict'])
        self.Q_target.load_state_dict(self.Q.state_dict())

    def evaluate_model(self, env, mode, op, num_episodes=50, max_steps=200):
        total_reward = 0
        for _ in range(num_episodes):
            ob, _info = env.reset(mode=mode)
            obs_agent2 = env.obs_agent_two()

            for t in range(max_steps):
                done = False      
                a1 = self.act(ob, eps=0)
                a2 = op.act(obs_agent2)
                (ob_new, reward, done, trunc, _info) = env.step(np.hstack([a1,a2]))
                ob=ob_new
                obs_agent2 = env.obs_agent_two()
                if done:
                    total_reward+= _info['winner']
                    break
        avg_reward = total_reward / num_episodes
        return avg_reward

class DuelingFeedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """ 
        Dueling DQN Network with Value and Advantage streams.
        
        input_size: Number of input features (dimensionality of the state).
        hidden_sizes: List of hidden layer sizes.
        output_size: Number of actions.
        """
        super(DuelingFeedforward, self).__init__()

        # Setting internal sizes for clarity
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Create shared hidden layers
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([torch.nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [torch.nn.ReLU() for _ in self.layers]

        # Additional hidden layer for the value stream
        self.value_hidden_layer = torch.nn.Linear(self.hidden_sizes[-1], 64)
        self.value_activation = torch.nn.ReLU()
        
        # Dueling architecture: Separate value and advantage streams
        self.value_layer = torch.nn.Linear(64, 1)  # State value (V(s))

        # Additional hidden layer for the advantage stream
        self.advantage_hidden_layer = torch.nn.Linear(self.hidden_sizes[-1], 64)
        self.advantage_activation = torch.nn.ReLU()
        
        self.advantage_layer = torch.nn.Linear(64, self.output_size)  # Advantage values (A(s,a))

    def forward(self, x):
        for layer, activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))

        # Passing through the additional hidden layer for the value stream
        value_hidden = self.value_activation(self.value_hidden_layer(x))
        value = self.value_layer(value_hidden)

        # Passing through the additional hidden layer for the advantage stream
        advantage_hidden = self.advantage_activation(self.advantage_hidden_layer(x))
        advantage = self.advantage_layer(advantage_hidden)
        
        # Combine value and advantage to get Q-values
        # Use the formula: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=0, keepdim=True))
        return q_values

    def predict(self, x):
        """ 
        Returns Q-values without gradients, mostly used during inference.
        
        x: Input array representing state(s).
        """
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()


class QFunction(DuelingFeedforward):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[256,256], learning_rate=0.00008):
        """ 
        Dueling DQN architecture with separate value and advantage streams.
        
        observation_dim: Dimensionality of the observation space.
        action_dim: Dimensionality of the action space.
        hidden_sizes: List of hidden layer sizes.
        learning_rate: Learning rate for the optimizer.
        """
        super().__init__(input_size=observation_dim, hidden_sizes=hidden_sizes, output_size=action_dim)
        
        # Adam optimizer with an epsilon for numerical stability
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=5e-6)
        
        # Huber Loss (SmoothL1Loss) is more robust to outliers than MSE.
        self.loss = torch.nn.SmoothL1Loss(reduction='none')

    def fit(self, observations, actions, targets):
        """ 
        Training routine for the Q-network. 
        
        observations: The observed states.
        actions: Actions taken.
        targets: TD-targets for those actions.
        """
        self.train()  # PyTorch: set the model in training mode
        self.optimizer.zero_grad()
        
        # Convert actions to their respective indices
        actions_indices = np.apply_along_axis(lambda x: action_to_index(x), 1, actions)
        Q_predictions = self.Q_value(torch.tensor(observations).float(), torch.tensor(actions_indices))

        # Compute the Huber Loss between predicted Q-values and the targets.
        loss_val = self.loss(Q_predictions, torch.tensor(targets).float())
        
        loss_val.mean().backward()
        self.optimizer.step()
        
        return np.squeeze(loss_val.tolist())

    def Q_value(self, observations, actions):
        """ 
        Return Q-values for the given state-action pairs.
        
        observations: The observed states.
        actions: Actions for which Q-values are required.
        """
        # Forward pass through the network and gather Q-values for the actions taken.
        return self.forward(observations).gather(1, actions[:, None])

    def maxQ(self, observations):
        """ 
        Return the maximum Q-value across actions for given observations. 
        Useful for computing the Q-learning targets.
        """
        return np.max(self.predict(observations), axis=-1, keepdims=True)

    def greedyAction(self, observations):
        """ 
        Use the Q-network to select the best action for a given state. 
        This forms the policy of the DQN agent.
        """
        return index_to_action(np.argmax(self.predict(observations), axis=-1))


def index_to_action(index):
    """Convert action index to 4-dimensional action."""
    assert 0 <= index < 16, "Index out of range!"
    
    # Using a binary-like representation
    action = [0, 0, 0, 0]
    action[0] = (index % 2) * 2 - 1  # -1 if remainder is 0, 1 if remainder is 1
    index //= 2
    action[1] = (index % 2) * 2 - 1
    index //= 2
    action[2] = (index % 2) * 2 - 1
    index //= 2
    action[3] = (index % 2) * 2 - 1
    
    return np.array(action, dtype=float)

def action_to_index(action):
    """Convert 4-dimensional action to action index."""
    assert len(action) == 4
    
    index = ((action[3] + 1) // 2) * 8 + ((action[2] + 1) // 2) * 4 + ((action[1] + 1) // 2) * 2 + (action[0] + 1) // 2
    
    return int(index)


class PrioritizedMemory():
    def __init__(self, max_size, alpha=0.6, beta=0.4, beta_increment=0.001):
        """ 
        Replay memory with prioritized sampling.

        max_size: Max number of transitions to store in the buffer.
        alpha: Prioritization level (0=uniform, 1=fully prioritized).
        beta: Corrective importance sampling factor.
        beta_increment: Increase rate of beta towards 1.
        """
        
        self.transitions = np.asarray([])  # Array to hold transitions.
        self.size = 0
        self.current_idx = 0
        self.max_size = max_size
        
        # Initial priorities for all transitions.
        self.priorities = np.zeros(max_size) + 1.0  
        self.alpha = alpha  
        self.beta = beta
        self.beta_increment = beta_increment 

    def _get_max_priority(self):
        """ Return max priority or 1.0 if empty. """
        return self.priorities.max() if self.size > 0 else 1.0

    def add_transition(self, transitions_new):
        """ Add a new transition to the buffer. """
        
        self.priorities[self.current_idx] = self._get_max_priority()
        
        # Initialize buffer if empty.
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx, :] = np.asarray(transitions_new, dtype=object)
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch=1):
        """ 
        Sample a batch of transitions based on their priorities.

        batch: Number of samples.
        """
        
        if batch > self.size:
            batch = self.size
        
        # Probabilities proportional to priorities.
        probs = (self.priorities[:self.size] ** self.alpha) / (self.priorities[:self.size] ** self.alpha).sum()
        self.inds = np.random.choice(range(self.size), size=batch, p=probs, replace=False)
        samples = self.transitions[self.inds, :]

        # Importance sampling weights
        max_weight = ((1 / self.size * 1 / self.priorities.min()) ** self.beta)
        weights = ((1 / self.size * 1 / self.priorities[self.inds]) ** self.beta) / max_weight
        weights = np.array(weights, dtype=np.float32)

        # Increment beta.
        self.beta = np.min([1., self.beta + self.beta_increment])

        return samples, weights

    def update_priorities(self, td_errors):
        """ Update priorities using TD-errors. """
        
        # Incremental priority to avoid zeros.
        for idx, error in zip(self.inds, td_errors):
            self.priorities[idx] = abs(error) + 1e-5