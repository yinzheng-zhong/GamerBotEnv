import collections

import tensorflow as tf
from src.Model.NN.neural_nets import NeuralNetwork
from src.Helper.configs import NN
import src.Helper.constance as constance
from src.Helper.configs import NN as nn_config
from src.Helper.configs import Keys as key_config


class Agent:
    def __init__(self, input_queue):
        self.input_queue = input_queue

        self.input_screen_dim = (
            nn_config.get_screenshot_input_dim()[1],
            nn_config.get_screenshot_input_dim()[0],
            3
        )  # (720, 1280)

        self.input_sound_dim = (
            constance.NN_SOUND_SPECT_INPUT_DIM[1],
            constance.NN_SOUND_SPECT_INPUT_DIM[0],
            1
        )  # (128, 128)
        self.key_output_size = key_config.get_key_mapping_size()

        self.replay_memory = collections.deque(maxlen=10000)

        self.pol_model = None
        self.tgt_model = None

    def gen_data(self):
        while True:
            data = self.input_queue.get()
            yield data['x'], data['y']

    def run(self):

        self.pol_model = NeuralNetwork().model

        o_type = (
            (tf.float32, tf.float32, tf.float32, tf.int8, tf.float32),
            (tf.int8, tf.float32)
        )

        dataset = tf.data.Dataset.from_generator(self.gen_data, output_types=o_type)
        #dataset = dataset.shuffle(buffer_size=constance.NN_SHUFFLE_BUFFER_SIZE)
        dataset = dataset.batch(NN.get_batch_size())

        self.pol_model.fit(dataset, epochs=1)
