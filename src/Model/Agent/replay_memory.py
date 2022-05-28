import numpy as np
from src.Helper.configs import *
import collections

MAX_SIZE = 10000


class ReplayMemory:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

        self._memory = collections.deque(maxlen=MAX_SIZE)

    def sample(self):
        choices = np.random.choice(len(self._memory), min(self.batch_size, len(self._memory)), replace=False)
        return [self._memory[i] for i in choices]

    def add(self, state, action, reward, new_state):
        self._memory.append((state, action, reward, new_state))
