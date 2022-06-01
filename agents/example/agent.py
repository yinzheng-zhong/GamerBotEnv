import numpy as np
import torch
from torch import optim

from src.Model.Agent.agent import Agent as AgentBase

from agents.example.nn import NN

""" Agent will take the last key """


class Agent(AgentBase):
    def __init__(self, in_queue, out_queue):
        super().__init__(in_queue, out_queue)

        self.main_model = NN().to(self.device)
        self.target_model = NN().eval().to(self.device)

        self.optimizer = optim.Adam(self.main_model.parameters(), lr=0.01)

        self.last_action = np.zeros(self.key_output_size, dtype=np.float32)

    def get_data(self):
        """
        Get the data from the queue.
        :return:
        """
        data = self._input_queue.get()
        state_with_actions = (*data['state'][:3], self.last_action)
        data['state'] = state_with_actions

        return data

    def action(self, action=None):
        """
        action=None will default to idle action. The action is a one-hot vector from the prediction, not the key string.
        :param action:
        :return state, reward, last_action:
        """
        if action is not None:
            self._action_out_queue.put(action)

        data = self.get_data()
        self.last_action = action

        return data['state'], data['reward']
