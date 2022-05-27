import collections
import time

import numpy as np
import tensorflow as tf
from src.Model.NN.neural_nets import NeuralNetwork
from src.Helper.configs import NN
import src.Helper.constance as constance
from src.Helper.configs import NN as nn_config
from src.Helper.configs import Keys as key_config
from src.Helper.configs import Agent as agent_config
from src.Model.Agent.replay_memory import ReplayMemory
from src.Utils.key_mapping import KeyMapping

DISCOUNT = 0.99

"""
This Agent is NOT DQN. It's just a simple agent as a showcase that takes human actions and reward, and it learns from it.
I use this to test various things. Please implement your own agent in agent folder that inherits from this.
"""


class CustomCallback(tf.keras.callbacks.Callback):
    @staticmethod
    def on_train_end(logs=None):
        print(logs)


class Agent:
    def __init__(self, input_queue, action_out_queue):
        self._input_queue = input_queue
        self._action_out_queue = action_out_queue

        self._agent_control = agent_config.get_agent_control()
        self.epsilon = 1
        self.epsilon_decay = agent_config.get_epsilon_decay()
        self.epsilon_min = agent_config.get_epsilon_min()

        self.input_screen_dim = (
            nn_config.get_screenshot_input_dim()[1],
            nn_config.get_screenshot_input_dim()[0],
            1
        )  # (720, 1280)

        self.input_sound_dim = (
            constance.NN_SOUND_SPECT_INPUT_DIM[1],
            constance.NN_SOUND_SPECT_INPUT_DIM[0],
            1
        )  # (128, 128)
        self.key_output_size = key_config.get_key_mapping_size()

        self.replay_memory = ReplayMemory(NN.get_batch_size())
        self.prediction_type = (tf.float32, tf.float32, tf.float32, tf.int8)

        self.training_type = (
                (tf.float32, tf.float32, tf.float32, tf.int8, tf.float32),
                tf.int8
            )

        self.main_model = None
        self.target_model = None
        self.weight_file = '{}weights-{}.h5'.format(constance.PATH_NN_WEIGHTS, nn_config.get_model_type())

        self.counter = 0

        self.key_mapping = KeyMapping()

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
        zip_samples = zip(*samples)
        return [np.array(val) for val in zip_samples]

    def action(self, action=None):
        """
        action=None will default to idle action.
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

    def train(self):
        batch = self.replay_memory.sample()

        if batch is None:
            return

        current_states = self.batch_samples_muilt_input([transition[0] for transition in batch])
        current_q_array = self.main_model.predict(current_states)

        new_states = self.batch_samples_muilt_input([transition[3] for transition in batch])
        future_q_array = self.target_model.predict(new_states)

        x = self.batch_samples_muilt_input([transition[0] for transition in batch])
        y = []

        for index, (current_state, action, reward, new_state) in enumerate(batch):
            '''this future_q_array has a mouse position so we use the first element'''
            max_future_q = np.max(future_q_array[index])
            new_q = reward + DISCOUNT * max_future_q

            # Update Q value for given state
            #  for predicted not human: current_qs = current_q_array[0][index]
            current_qs = current_q_array[index]
            current_qs[np.argmax(action)] = new_q

            y.append(current_qs)

        self.main_model.fit(
            x=x,
            y=np.array(y),
            epochs=1,
            verbose=0,
            callbacks=[CustomCallback()] if self.counter % 10 == 0 else None
        )

        if not self.counter % 5:
            self.target_model.set_weights(self.main_model.get_weights())

    def run(self):
        print('Agent is running')
        human_correction_rate = 3

        self.main_model = NeuralNetwork().model
        self.target_model = NeuralNetwork().model

        try:
            self.target_model.load_weights(self.weight_file)
        except ValueError:
            print('Model params have changed')
        except FileNotFoundError:
            print('Model weight file not found')

        current_data = self.get_data()

        current_state = current_data['state'][:3]

        while True:
            if self._agent_control:
                if np.random.random() > self.epsilon:
                    predicted_value = self.main_model.predict(
                        self.batch_samples_muilt_input([current_state])
                    )[0]
                    action = np.argmax(predicted_value)
                else:
                    '''Action prediction is done be human'''
                    action_ind = np.random.randint(0, self.key_output_size)
                    action = np.zeros(self.key_output_size)
                    action[action_ind] = 1

                predicted_action = self.key_mapping.get_key_from_on_hot_mapping(action)

                print('\npredicted_action: {}'.format(predicted_action))

                new_state, reward = self.action(predicted_action)
            else:
                new_data = self.get_data()
                new_state = new_data['state'][:3]
                action = new_data['action']
                print('\naction: {}'.format(action))
                #  always give some reward to human actions
                reward = new_data['reward'] if new_data['reward'] != 0 else 4

            self.replay_memory.add(current_state, action, reward, new_state)

            current_state = new_state

            self.train()

            self.counter += 1

            if not self.counter % 100:
                self.counter = 0
                self.target_model.save_weights(self.weight_file)
