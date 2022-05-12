import numpy as np
import src.Utils.image as image_utils
import src.Utils.audio as audio_utils
import src.Utils.key_mapping as key_mapping
from src.Helper.config_reader import Capturing, Hardware, NN
from multiprocessing import Pool


class InputProcessing:
    def __init__(self, input_queue, output_queue):
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.audio_sample_rate = Capturing.get_audio_sample_rate()
        self.screen_resolution = Hardware.get_screen_res()
        self.nn_screenshot_input_dim = NN.get_screenshot_input_dim()

        self.key_mapper = key_mapping.KeyMapping()

    def run(self):
        while True:
            self.process_input()

    def process_input(self):
        data = self.input_queue.get(block=True)

        x = data['x']  # a batch of input values
        y = data['y']  # a batch of target values

        new_x = list(map(self.process_single_input, x))
        new_y = list(map(self.process_single_output, y))

        self.output_queue.put({'x': new_x, 'y': new_y})

        image_utils.save_image(new_x[0][1], 'test.png')
        print('batch processed and ready!!!')

    def process_single_input(self, data):
        image = data['screenshot']

        audio_l = data['audio_l']
        audio_r = data['audio_r']

        s_image = image_utils.scale_image(image, self.nn_screenshot_input_dim)

        norm_image = image_utils.image_normalise(s_image)
        mel_spectr_l = audio_utils.mel_spectrogram_mono(audio_l, self.audio_sample_rate)
        mel_spectr_r = audio_utils.mel_spectrogram_mono(audio_r, self.audio_sample_rate)

        return norm_image, mel_spectr_l, mel_spectr_r

    def process_single_output(self, data):
        key = data['action']
        cursor_pos = data['cursor']

        key_vec = self.key_mapper.get_on_hot_mapping(key)

        x_axis = cursor_pos[0] / self.screen_resolution[0]
        y_axis = cursor_pos[1] / self.screen_resolution[1]

        return key_vec, (x_axis, y_axis)
