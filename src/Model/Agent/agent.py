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
from src.Model.Agent.replay_memory import ReplayMemory
from src.Utils.key_mapping import KeyMapping
import src.Model.NN.nn_pytorch as nn_pytorch
import torch.nn as nn


DISCOUNT = 0.99

""" Migrated from Tensorflow. Need some optimisation later """


# class CustomCallback(tf.keras.callbacks.Callback):
#     @staticmethod
#     def on_train_end(logs=None):
#         print(logs)


class Agent:
    def __init__(self, input_queue, action_out_queue):
        self._input_queue = input_queue
        self._action_out_queue = action_out_queue

        self._agent_control = agent_config.get_agent_control()
        self.epsilon = agent_config.get_epsilon()
        self.epsilon_decay = agent_config.get_epsilon_decay()
        self.epsilon_min = agent_config.get_epsilon_min()

        self.key_output_size = key_config.get_key_mapping_size()

        self.replay_memory = ReplayMemory(NN.get_batch_size())
        # self.prediction_type = (tf.float32, tf.float32, tf.float32, tf.int8)
        #
        # self.training_type = (
        #         (tf.float32, tf.float32, tf.float32, tf.int8, tf.float32),
        #         tf.int8
        #     )

        self.device = torch.device("cuda:0")

        self.main_model = nn_pytorch.CNN().to(self.device)
        self.target_model = nn_pytorch.CNN().eval().to(self.device)

        self.optimizer = optim.Adam(self.main_model.parameters())
        self.loss_fn = nn.MSELoss()

        self.weight_file = '{}weights-{}.h5'.format(constance.PATH_NN_WEIGHTS, nn_config.get_model_type())

        self.counter = 0

        self.key_mapping = KeyMapping()

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
        :param action:
        :return state, reward:
        """
        if action is not None:
            self._action_out_queue.put(action)

        data = self.get_data()
        return data['state'], data['reward']

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
        batch = self.replay_memory.sample()

        if len(batch) < NN.get_batch_size():
            return

        current_states = self.batch_samples_muilt_input_tensors([transition[0] for transition in batch])
        # The states are already tensors before we put them into the replay memory.
        current_q_array = self.main_model(*current_states)

        new_states = self.batch_samples_muilt_input_tensors([transition[3] for transition in batch])
        with torch.no_grad():
            future_q_array = self.target_model(*new_states)

        y = []

        for index, (current_state, action, reward, new_state) in enumerate(batch):
            '''this future_q_array has a mouse position so we use the first element'''
            max_future_q = torch.max(future_q_array[index])
            new_q = reward + DISCOUNT * max_future_q

            # Update Q value for given state
            #  for predicted not human: current_qs = current_q_array[0][index]
            current_qs = current_q_array[index].clone().detach().requires_grad_(False)
            current_qs[torch.argmax(action)] = new_q

            y.append(current_qs)

        y = torch.stack(y)

        self.optimizer.zero_grad()
        loss = self.loss_fn(current_q_array, y)
        loss.backward()
        self.optimizer.step()

        if not self.counter % 5:
            self.target_model.load_state_dict(self.main_model.state_dict())

    def run(self):
        print('Agent is running')

        try:
            self.main_model.load_state_dict(torch.load(self.weight_file))
            self.target_model.load_state_dict(torch.load(self.weight_file))
        except (_pickle.UnpicklingError, RuntimeError):
            print('Model params have changed')
        except FileNotFoundError:
            print('Model weight file not found')

        current_data = self.get_data()

        current_state = current_data['state'][:3]
        current_state = self.state_np_to_device(current_state)

        while True:
            if self._agent_control:
                if np.random.random() > self.epsilon:
                    on_device_state = self.state_tensor_to_device(current_state)
                    with torch.no_grad():
                        action = self.main_model(*(x.unsqueeze(0) for x in on_device_state)).cpu()
                    numpy_action = action.numpy()
                else:
                    action_ind = np.random.randint(0, self.key_output_size)
                    action = np.zeros(self.key_output_size)
                    action[action_ind] = 1

                    numpy_action = np.array(action)
                    action = torch.tensor(action)

                predicted_action = self.key_mapping.get_key_from_on_hot_mapping(numpy_action)

                print('\npredicted_action: {}'.format(numpy_action))

                new_state, reward = self.action(numpy_action)
                new_state = new_state[:3]  # remove the feedback

                new_state = self.state_np_to_device(new_state)
                reward = torch.tensor(reward)

                self.update_epsilon()

            else:
                '''Action prediction is done be human'''
                new_data = self.get_data()
                new_state = new_data['state'][:3]

                action = new_data['action']
                #  always give some reward to human actions
                reward = new_data['reward'] if new_data['reward'] != 0 else 4

                # all to device
                new_state = self.state_np_to_device(new_state)
                action = torch.tensor(action)
                reward = torch.tensor(reward)

            self.replay_memory.add(current_state, action, reward, new_state)

            current_state = new_state

            self.train()

            self.counter += 1

            if not self.counter % 100:
                self.counter = 0
                torch.save(self.target_model.state_dict(), self.weight_file)
