from typing import List
import math
import random
from collections import namedtuple, deque
from itertools import count

import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

# Select the device
device = torch.device('cpu')

# Define a named tuple that represnets an experience
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Define the hyper parameters for the DDQN algorithm
BATCH_SIZE = 128
GAMMA = 0.99
# The initial porbability of uniformly selecting an action (not greedy)
EPS_START = 0.9
# The final probability of uniformly selecting an action (not greedy)
EPS_END = 0.05
# The decay size on the probability of uniformly selecting an action
EPS_DECAY = 1000
# The extent to which the target network parameters should be updated (taken from policy network)
TAU = 0.005
# Learning rate for optimization
LR = 1e-4
# Size of the memory
MEMORY_SIZE = 10000

# Cart-pole game
env = gym.make('CartPole-v1')

# Number of actions possible from each state
n_actions = env.action_space.n
# Number of dimensions (representation) for each state
state, info = env.reset()
dim_state = len(state)

# Define a class that represents the experience replay memory
class ReplayMemory(object):

    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        # Sampling uniformly at random (without replacement)
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

# Define the policy as well as the target network
class DQN(nn.Module):

    '''
    A fully connected neural network representing the policy (Q)
    and the target networks

    Parameters
    ==========
    dim_state: The dimensionality of the states
    n_actions: The number of possible actions from each state
    '''

    def __init__(self, dim_state: int, n_actions: int) -> None:
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(dim_state, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

def select_action(state: torch.Tensor) -> torch.Tensor:
    '''
    Performs epsilon-greedy action selection given a state
    '''
    # The steps_done is in the global scope
    global steps_done
    # Generate randomly a number between 0 and 1 
    sample = random.random()
    # Probability for the epsilon-greedy selection
    # It decays as the number of steps grow
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1 * steps_done / EPS_DECAY)
    steps_done += 1
    # Returns a tensor of dim (1, 1) (tensor([[1]]), for example)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) (t stands for a tensor) returns a tensor
            # where the first column is the largest column found for 
            # each row in t and the second column is the index
            # of the column at which the maximum value happened. 
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    '''
    Double DQN (DDQN) implementation (Hasselt et al, 2015)
    '''
    # No update if the buffer is not adequately populated
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    # A boolean tensor (true if the next state is not None and false, otherwise)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype=torch.bool)
    # Concat only next states that are not None (with a non zero Q)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                 if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Get the action values (Q) for each state and the action chosen
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        # Get the action for the next state whose expected value
        # evaluated by the policy network is maximum (argmax Q(next_state, action; θ))
        next_state_actions = policy_net(non_final_next_states).max(1).indices.view(-1, 1)
        # Get the value of the action (selected above) using
        # the target network (Q(next_state, action; θ'))
        next_state_values[non_final_mask] = target_net(non_final_next_states) \
            .gather(1, next_state_actions).squeeze(1)
    
    # Get the target values for the policy network (needed for optimization)
    expected_state_action_values = reward_batch + GAMMA * next_state_values

    # Huber loss
    criterion = nn.SmoothL1Loss()
    # unsqueeze adds dim 1 into the requested axis (here, the tensor is reshaped into a column vector)
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    # Apply the gradient clipping
    nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    # Update the parameters of the policy network
    optimizer.step()

# Define the policy and the target networks
policy_net = DQN(dim_state, n_actions).to(device)
# Targets for the policy net are provided by the target net
target_net = DQN(dim_state, n_actions).to(device)
# Copy the parameters from the policy net into the target net
target_net.load_state_dict(policy_net.state_dict()) 

# Optimizer
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
# Buffer using which the experiences are stored and
# sampled (uniformly at random) later
memory = ReplayMemory(MEMORY_SIZE)

# Number of steps (t) taken so far
steps_done = 0

# Number of episodes (playing the game)
num_episodes = 600

for episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        # Get the action using the current policy network
        action = select_action(state)
        # Get the next state, reward, status of the next state, ...
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated
        # If the next state is a terminal state, then set it to None
        if terminated:
            next_state = None
        else:
            # A row vector of shape (1, 4)
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        # Add the experience to the buffer 
        memory.push(state, action, next_state, reward)
        # Move to the next state
        state = next_state
        # Sample from the buffer and optimize the policy network parameters  
        optimize_model()
        # Copy the parameters of the policy network to the target network
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        # Update the parameters of the target network
        target_net.load_state_dict(target_net_state_dict)
        if done:
            # End of the episode
            break