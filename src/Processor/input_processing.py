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

        self.windows = {}
        self.time_steps = NN.get_time_steps()

    def run(self):
        print('Input preprocessing started')
        if NN.get_model_type() == constance.NN_MODEL_SINGLE:
            while True:
                data = self.process_data()

                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except q.Empty:
                        pass

                self.output_queue.put(data)
        else:
            while True:
                self.init_window()
                self.update_window()

    def init_window(self):
        """
        Initialize the sliding window for LSTM
        """
        screenshot = []
        action = []
        mouse = []

        audio_l = None
        audio_r = None

        action_out = None
        mouse_out = None

        reward = None

        for i in range(self.time_steps):
            data = self.process_data()
            screenshot.append(data['state'][0])
            action.append(data['state'][3])
            mouse.append(data['state'][4])

            if i == self.time_steps - 1:
                audio_l = data['state'][1]
                audio_r = data['state'][2]

                action_out = data['action'][0]
                mouse_out = data['action'][1]

                reward = data['reward']

        self.windows = {'screenshot': screenshot, 'action': action, 'mouse': mouse}

        self.output_queue.put({
            'state': (
                np.asarray(screenshot, dtype=np.float32),
                audio_l,
                audio_r,
                np.asarray(action, dtype=np.int8),
                np.asarray(mouse, dtype=np.float32)
            ),
            'action': (action_out, mouse_out),
            'reward': reward
        })

    def update_window(self):
        data = self.process_data()

        # screenshot
        self.windows['screenshot'].pop(0)
        self.windows['screenshot'].append(data['state'][0])

        # action
        self.windows['action'].pop(0)
        self.windows['action'].append(data['state'][3])

        # mouse
        self.windows['mouse'].pop(0)
        self.windows['mouse'].append(data['state'][4])

        self.output_queue.put({
            'state': (
                np.asarray(self.windows['screenshot'], dtype=np.float32),
                data['state'][1],
                data['state'][2],
                np.asarray(self.windows['action'], dtype=np.int8),
                np.asarray(self.windows['mouse'], dtype=np.float32)
            ),
            'action': (data['action'][0], data['action'][1]),
            'reward': data['reward']
        })

    def process_data(self):
        data = self.input_queue.get()

        state = data['state']  # a batch of input values
        action = data['action']  # a batch of target values

        #new_x = list(map(self.process_single_input, state))
        #new_y = list(map(self.process_single_output, action))

        new_x = self._process_single_state(state)
        new_y = self._process_single_action(action)

        self.last_action = new_y[0]
        self.last_mouse = new_y[1]

        if time.time() - self.last_check > 20:
            self.last_check = time.time()
            print('\033[93m\nProcessing queue has {} items left.\033[0m'.format(self.input_queue.qsize()))
            print('\033[93m\nTraining queue has {} items left.\033[0m'.format(self.output_queue.qsize()))

        return {'state': (new_x[0], new_x[1], new_x[2], self.last_action, self.last_mouse),
                'action': (new_y[0], new_y[1]),
                'reward': data['reward']}

    def _process_single_state(self, data):
        image = data['screenshot']

        audio_l = data['audio_l']
        audio_r = data['audio_r']

        gray_image = image_utils.convert_to_grayscale(image)
        s_image = image_utils.scale_image(gray_image, self.nn_screenshot_input_dim)

        norm_image = image_utils.image_normalise(s_image)
        mel_spectr_l = audio_utils.mel_spectrogram_mono(audio_l, self.audio_sample_rate)
        mel_spectr_r = audio_utils.mel_spectrogram_mono(audio_r, self.audio_sample_rate)

        # normalised spectrogram and scale.
        norm_mel_spectr_l = image_utils.image_normalise(mel_spectr_l)
        norm_mel_spectr_r = image_utils.image_normalise(mel_spectr_r)

        s_mel_spectr_l = image_utils.scale_image(norm_mel_spectr_l, constance.NN_SOUND_SPECT_INPUT_DIM)
        s_mel_spectr_r = image_utils.scale_image(norm_mel_spectr_r, constance.NN_SOUND_SPECT_INPUT_DIM)

        reshape_image = np.expand_dims(norm_image, axis=-1)
        reshape_mel_spectr_l = np.expand_dims(s_mel_spectr_l, axis=-1)
        reshape_mel_spectr_r = np.expand_dims(s_mel_spectr_r, axis=-1)

        return [reshape_image, reshape_mel_spectr_l, reshape_mel_spectr_r]

    def _process_single_action(self, data):
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
