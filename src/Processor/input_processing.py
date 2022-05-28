import numpy as np
import src.Utils.image as image_utils
import src.Utils.audio as audio_utils
import src.Utils.key_mapping as key_mapping
from src.Helper.configs import Capturing, Hardware, NN, Agent
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

        self.agent_control = Agent.get_agent_control()

        self.last_action = self.key_mapper.get_on_hot_mapping(None)  # feedback loop
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

        audio_l = None
        audio_r = None

        action_out = None

        reward = None

        for i in range(self.time_steps):
            data = self.process_data()
            screenshot.append(data['state'][0])
            action.append(data['state'][3])

            if i == self.time_steps - 1:
                audio_l = data['state'][1]
                audio_r = data['state'][2]

                action_out = data['action']

                reward = data['reward']

        self.windows = {'screenshot': screenshot, 'action': action}

        self.output_queue.put({
            'state': (
                np.asarray(screenshot, dtype=np.float32),
                audio_l,
                audio_r,
                np.asarray(action, dtype=np.int8)
            ),
            'action': action_out,
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

        self.output_queue.put({
            'state': (
                np.asarray(self.windows['screenshot'], dtype=np.float32),
                data['state'][1],
                data['state'][2],
                np.asarray(self.windows['action'], dtype=np.int8),
            ),
            'action': data['action'],
            'reward': data['reward']
        })

    def process_data(self):
        data = self.input_queue.get()

        state = data['state']  # a batch of input values
        action = data['action']  # a batch of target values

        #new_x = list(map(self.process_single_input, state))
        #new_y = list(map(self.process_single_output, action))

        new_x = self._process_single_state(state)
        new_y = self._process_single_action(action) if self.agent_control else None

        self.last_action = new_y

        if time.time() - self.last_check > 20:
            self.last_check = time.time()
            print('\033[93m\nProcessing queue has {} items left.\033[0m'.format(self.input_queue.qsize()))
            print('\033[93m\nTraining queue has {} items left.\033[0m'.format(self.output_queue.qsize()))

        return {'state': (new_x[0], new_x[1], new_x[2], self.last_action),
                'action': new_y,
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

        reshape_image = np.moveaxis(reshape_image, -1, 0)
        reshape_mel_spectr_l = np.moveaxis(reshape_mel_spectr_l, -1, 0)
        reshape_mel_spectr_r = np.moveaxis(reshape_mel_spectr_r, -1, 0)

        return [reshape_image, reshape_mel_spectr_l, reshape_mel_spectr_r]

    def _process_single_action(self, key):
        key_vec = self.key_mapper.get_on_hot_mapping(key)

        return key_vec
