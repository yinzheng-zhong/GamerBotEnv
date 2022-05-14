import numpy as np
import src.Utils.image as image_utils
import src.Utils.audio as audio_utils
import src.Utils.key_mapping as key_mapping
from src.Helper.configs import Capturing, Hardware, NN
import src.Helper.constance as constance
from multiprocessing import Pool
import time
import queue as q


class Preprocessing:
    def __init__(self, input_queue, output_queue):
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.audio_sample_rate = Capturing.get_audio_sample_rate()
        self.screen_resolution = Hardware.get_screen_res()
        self.nn_screenshot_input_dim = NN.get_screenshot_input_dim()

        self.key_mapper = key_mapping.KeyMapping()

        self.last_action = self.key_mapper.get_on_hot_mapping(None)  # feedback loop
        self.last_mouse = np.array((0, 0))
        self.last_check = 0

    def run(self):
        while True:
            data = self.process_data()

            if self.output_queue.full():
                try:
                    self.output_queue.get_nowait()
                except q.Empty:
                    pass

            self.output_queue.put(data)

    def process_data(self):
        data = self.input_queue.get()

        x = data['x']  # a batch of input values
        y = data['y']  # a batch of target values

        #new_x = list(map(self.process_single_input, x))
        #new_y = list(map(self.process_single_output, y))

        new_x = self.process_single_input(x)
        new_y = self.process_single_output(y)

        self.last_action = new_y[0]
        self.last_mouse = new_y[1]

        if time.time() - self.last_check > 20:
            self.last_check = time.time()
            print('Processing queue has {} items left.'.format(self.input_queue.qsize()))
            print('Training queue has {} items left.'.format(self.output_queue.qsize()))

        return {'x': (new_x[0], new_x[1], new_x[2], self.last_action, self.last_mouse),
                'y': (new_y[0], new_y[1])}

    def process_single_input(self, data):
        image = data['screenshot']

        audio_l = data['audio_l']
        audio_r = data['audio_r']

        s_image = image_utils.scale_image(image, self.nn_screenshot_input_dim)

        norm_image = image_utils.image_normalise(s_image)
        mel_spectr_l = audio_utils.mel_spectrogram_mono(audio_l, self.audio_sample_rate)
        mel_spectr_r = audio_utils.mel_spectrogram_mono(audio_r, self.audio_sample_rate)

        # normalised spectrogram and scale.
        norm_mel_spectr_l = image_utils.image_normalise(mel_spectr_l)
        norm_mel_spectr_r = image_utils.image_normalise(mel_spectr_r)

        s_mel_spectr_l = image_utils.scale_image(norm_mel_spectr_l, constance.NN_SOUND_SPECT_INPUT_DIM)
        s_mel_spectr_r = image_utils.scale_image(norm_mel_spectr_r, constance.NN_SOUND_SPECT_INPUT_DIM)

        reshape_mel_spectr_l = np.expand_dims(s_mel_spectr_l, axis=-1)
        reshape_mel_spectr_r = np.expand_dims(s_mel_spectr_r, axis=-1)

        return [norm_image, reshape_mel_spectr_l, reshape_mel_spectr_r]

    def process_single_output(self, data):
        key = data['action']
        cursor_pos = data['cursor']

        key_vec = self.key_mapper.get_on_hot_mapping(key)

        x_axis = cursor_pos[0] / self.screen_resolution[0]
        if x_axis > 1:
            x_axis = 1

        y_axis = cursor_pos[1] / self.screen_resolution[1]
        if y_axis > 1:
            y_axis = 1

        return [key_vec, np.asarray((x_axis, y_axis), dtype=np.float32)]
