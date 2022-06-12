import _pickle
import collections
import time

import numpy as np
#import tensorflow as tf
# from src.Model.NN.nn_keras import NeuralNetwork
import torch
from torch import optim

from src.Helper.configs import NN
import src.Helper.constance as constance
from src.Helper.configs import NN as nn_config
from src.Helper.configs import Keys as key_config
from src.Helper.configs import Agent as agent_config
from src.Helper.configs import Hardware as hardware_config
from src.Model.Agent.replay_memory import ReplayMemory
from src.Utils.key_mapping import KeyMapping
import src.Model.NN.nn_pytorch as nn_pytorch
import torch.nn as nn
from torch.cuda.amp import autocast

""" Migrated from Tensorflow. Need some optimisation later """


class AgentBase:
    def __init__(self, input_queue, action_out_queue):
        self._input_queue = input_queue
        self._action_out_queue = action_out_queue

        self._agent_control = agent_config.get_agent_control()
        self.epsilon = agent_config.get_epsilon()
        self.epsilon_decay = agent_config.get_epsilon_decay()
        self.epsilon_min = agent_config.get_epsilon_min()

        self.key_output_size = key_config.get_key_mapping_size()
        self.key_mapping = KeyMapping()

        self.replay_memory = ReplayMemory(NN.get_batch_size())
        # self.prediction_type = (tf.float32, tf.float32, tf.float32, tf.int8)
        #
        # self.training_type = (
        #         (tf.float32, tf.float32, tf.float32, tf.int8, tf.float32),
        #         tf.int8
        #     )

        self.device = torch.device(hardware_config.get_device() if torch.cuda.is_available() else "cpu")
        self.gamma = torch.tensor(agent_config.get_gamma()).to(self.device)

        self.main_model = None
        self.target_model = None

        self.optimizer = None

        self.weight_file = '{}weights-{}.h5'.format(constance.PATH_NN_WEIGHTS, nn_config.get_model_type())

    def state_tensor_to_device(self, state):
        """
        State tensors to device with multiple inputs
        """
        return tuple(x.to(self.device) for x in state)

    def state_np_to_tensors(self, state):
        return tuple(torch.tensor(x) for x in state)

    def state_np_to_device(self, state):
        """
        State to device with multiple inputs
        """
        data_on_dev = tuple(torch.tensor(x).to(self.device) for x in state)
        return data_on_dev

    def gen_data(self):
        """
        This can be connected to the TF Data dataset API.
        :return:
        """
        while True:
            data = self._input_queue.get()
            yield data['state'], data['action']

    def batch_samples_muilt_input(self, samples):
        """
        This is used to batch samples for neural_nets. Used when you have multiple inputs or outputs.
        :param samples:
        :return:
        """
        zip_samples = list(zip(*samples))

        numpy_sample = [np.array(sample) for sample in zip_samples]
        return numpy_sample

    def batch_samples_muilt_input_tensors(self, samples):
        """
        This is used to batch samples for neural_nets. Used when you have multiple inputs or outputs.
        :param samples:
        :return:
        """

        data = list(zip(*samples))
        return tuple(torch.stack(x) for x in data)

    def action(self, action=None):
        """
        action=None will default to idle action. The action is a one-hot vector from the prediction, not the key string.
        By default only the screenshot and the audio channels are returned in state.
        :param action:
        :return state, reward:
        """
        # if action is not None:
        #     self._action_out_queue.put(action)

        data = self.get_data()
        return data['state'][:3], data['reward']

    def get_data(self):
        """
        This is used to get data from the queue when you don't need the agent actions. Normally for reinforcement learning.
        :return {state, action, reward}:
        """
        return self._input_queue.get()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    def train(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
