import collections

import numpy as np
import tensorflow as tf
from src.Model.NN.neural_nets import NeuralNetwork
from src.Helper.configs import NN
import src.Helper.constance as constance
from src.Helper.configs import NN as nn_config
from src.Helper.configs import Keys as key_config
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
    def __init__(self, input_queue):
        self.input_queue = input_queue

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
        self.prediction_type = (tf.float32, tf.float32, tf.float32, tf.int8, tf.float32)

        self.training_type = (
                (tf.float32, tf.float32, tf.float32, tf.int8, tf.float32),
                (tf.int8, tf.float32)
            )

        self.model = None
        self.weight_file = '{}weights-{}.h5'.format(constance.PATH_NN_WEIGHTS, nn_config.get_model_type())

        self.counter = 0
        self.got_mouse_cursor = False  # seems the mouse cursor comes very late. so we need to wait for it.

        self.key_mapping = KeyMapping()

    def gen_data(self):
        while True:
            data = self.input_queue.get()
            yield data['state'], data['action']

    def batch_input_samples(self, samples):
        data = {'x1': np.array([sample[0] for sample in samples]), 'x2': np.array([sample[1] for sample in samples]),
                'x3': np.array([sample[2] for sample in samples]), 'x4': np.array([sample[3] for sample in samples]),
                'x5': np.array([sample[4] for sample in samples])}

        return data

    def batch_output_samples(self, samples):
        data = {'y1': np.array([sample[0] for sample in samples]), 'y2': np.array([sample[1] for sample in samples])}

        return data

    def train(self):
        batch = self.replay_memory.sample()

        if batch is None:
            return

        #current_states = [transition[0] for transition in batch]
        #dataset = Agent.batch_input_samples(current_states)
        current_q_array = np.array([transition[1][0] for transition in batch])
        current_mouse = np.array([transition[1][1] for transition in batch])

        new_states = [transition[3] for transition in batch]
        dataset = self.batch_input_samples(new_states)
        future_q_array = self.model.predict(dataset)

        x = []
        y = []

        for index, (current_state, action, reward, new_state) in enumerate(batch):
            '''this future_q_array has a mouse position so we use the first element'''
            max_future_q = np.max(future_q_array[0][index])
            new_q = reward + DISCOUNT * max_future_q

            # Update Q value for given state
            #  for predicted not human: current_qs = current_q_array[0][index]
            current_qs = current_q_array[index]
            current_qs[np.argmax(action[0])] = new_q

            x.append(current_state)
            y.append([current_qs, current_mouse[index]])

        self.model.fit(
            x=self.batch_input_samples(x),
            y=self.batch_output_samples(y),
            epochs=1,
            verbose=0,
            callbacks=[CustomCallback()] if self.counter % 10 == 0 else None
        )

    def run(self):
        print('Agent is running')
        self.model = NeuralNetwork().model

        try:
            self.model.load_weights(self.weight_file)
        except ValueError:
            print('Model params have changed')
        except FileNotFoundError:
            print('Model weight file not found')

        current_data = self.input_queue.get()

        current_state = current_data['state']
        current_action = current_data['action']

        while True:
            predicted_action = self.model.predict(self.batch_input_samples([current_state]))
            print('\npredicted_action: {}'.format(self.key_mapping.get_key_from_on_hot_mapping(predicted_action[0][0])))

            '''Action prediction is done be human'''
            new_data = self.input_queue.get()

            new_state = new_data['state']
            new_action = new_data['action']

            if not self.got_mouse_cursor:
                if sum(new_action[1]):
                    self.got_mouse_cursor = True
                else:
                    continue

            reward = new_data['reward']

            self.replay_memory.add(current_state, current_action, reward, new_state)

            current_state = new_state
            current_action = new_action

            self.train()

            self.counter += 1

            if not self.counter % 100:
                self.counter = 0
                self.model.save_weights(self.weight_file)
