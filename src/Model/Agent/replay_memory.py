import numpy as np
from src.Helper.configs import *
import collections

MAX_SIZE = 10000


class ReplayMemory:
    def __init__(self, input_shape):
        self.mem_cntr = 0
        self.state_memory = collections.deque(maxlen=MAX_SIZE)
        self.action_memory = collections.deque(maxlen=MAX_SIZE)
        self.reward_memory = collections.deque(maxlen=MAX_SIZE)

        self.new_state_memory = collections.deque(maxlen=MAX_SIZE)

    def sample_buffer(self, batch_size):
        max_samples = min(self.mem_cntr, MAX_SIZE)
        batch = np.random.choice(max_samples, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]

        return states, actions, rewards, states_,
