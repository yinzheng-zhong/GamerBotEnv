from etc.settings import Settings
import cv2
import pyaudio
import src.Helper.constance as constance


class NN:
    @staticmethod
    def get_time_steps():
        data = 10
        try:
            data = Settings.neural_network['time_steps']
        except KeyError:
            print('[ERROR] Neural network time steps not found in config file.')

        return data

    @staticmethod
    def get_batch_size():
        data = 64
        try:
            data = Settings.neural_network['batch_size']
        except KeyError:
            print('[ERROR] Neural network batch size not found in config file.')

        return data

    @staticmethod
    def get_screenshot_input_dim():
        data = (1280, 720)
        try:
            data = Settings.neural_network['screenshot_input_dim']
        except KeyError:
            print('[ERROR] Neural network screenshot input dim not found in config file.')

        return data

    @staticmethod
    def get_model_type():
        data = constance.NN_MODEL_SINGLE
        try:
            data = Settings.neural_network['model']
        except KeyError:
            print('[ERROR] Neural network model type not found in config file.')

        return data

    @staticmethod
    def get_training_queue_size():
        data = 1000
        try:
            data = Settings.neural_network['training_queue_size']
        except KeyError:
            print('[ERROR] Neural network training queue size not found in config file.')

        return data


class Agent:
    @staticmethod
    def get_reward_time_gap():
        data = 5
        try:
            data = Settings.agent['reward_time_gap']
        except KeyError:
            print('[ERROR] Agent reward gap not found in config file.')

        return data

    @staticmethod
    def get_agent_control():
        data = True
        try:
            data = Settings.agent['agent_control']
        except KeyError:
            print('[ERROR] Agent control not found in config file.')

        return data

    @staticmethod
    def get_default_reward():
        data = 0
        try:
            data = Settings.agent['default_reward']
        except KeyError:
            print('[ERROR] Agent default reward not found in config file.')

        return data

    @staticmethod
    def get_gamma():
        data = 0.9
        try:
            data = Settings.agent['gamma']
        except KeyError:
            print('[ERROR] Agent gamma not found in config file.')

        return data

    @staticmethod
    def get_epsilon():
        data = 1
        try:
            data = Settings.agent['epsilon']
        except KeyError:
            print('[ERROR] Agent epsilon not found in config file.')

        return data

    @staticmethod
    def get_epsilon_decay():
        data = 0.9999
        try:
            data = Settings.agent['epsilon_decay']
        except KeyError:
            print('[ERROR] Agent epsilon decay not found in config file.')

        return data

    @staticmethod
    def get_agent_class_path_name():
        data = ('src.Agent.Agent', 'Agent')
        try:
            data = Settings.agent['agent_class_path_name']
        except KeyError:
            print('[ERROR] Agent class path name not found in config file.')

        return data

    @staticmethod
    def get_epsilon_min():
        data = 0.01
        try:
            data = Settings.agent['epsilon_min']
        except KeyError:
            print('[ERROR] Agent epsilon min not found in config file.')

        return data


class Capturing:
    @staticmethod
    def get_frame_rate():
        data = 0
        try:
            data = Settings.capturing['video_frame_rate']
        except KeyError:
            print('[ERROR] Capturing frame rate not found in config file.')

        return data

    @staticmethod
    def get_resolution():
        data = (0, 0)

        try:
            data = Settings.capturing['video_resolution']
        except KeyError:
            print('[ERROR] Capturing resolution not found in config file. Default resolution is 512x512.')

        return data

    @staticmethod
    def get_audio_format():
        data = pyaudio.paInt16
        try:
            data = Settings.capturing['audio_format']
        except KeyError:
            print('[ERROR] Capturing audio bit not found in config file. Default audio bit is 16.')

        return data

    @staticmethod
    def get_audio_sample_rate():
        data = 44100
        try:
            data = Settings.capturing['audio_sample_rate']
        except KeyError:
            print('[ERROR] Capturing audio sample rate not found in config file. Default audio sample rate is 44100.')

        return data

    @staticmethod
    def get_audio_length():
        data = 2
        try:
            data = Settings.capturing['audio_length']
        except KeyError:
            print('[ERROR] Capturing audio length not found in config file. Default audio length is 2.')

        return data


class Hardware:
    @staticmethod
    def get_screen_res():
        data = (0, 0)

        try:
            data = Settings.hardware['screen_resolution']
        except KeyError:
            print('[ERROR] Hardware screen resolution not found in config file.')

        if not all(data):
            raise ValueError('[ERROR] Hardware screen resolution not found in config file.')

        return data

    @staticmethod
    def get_audio_mixer_id():
        data = -1
        try:
            data = Settings.hardware['audio_stereo_mixer_device_id']
        except KeyError:
            print('[ERROR] Capturing audio mixer id not found in config file. Default audio mixer id is -1.')

        return data

    @staticmethod
    def get_device():
        data = 'cpu'
        try:
            data = Settings.hardware['use_device']
        except KeyError:
            print('[ERROR] Hardware gpu id not found in config file. Default device is cpu.')

        return data


class TM:
    @staticmethod
    def get_method():
        data = cv2.TM_CCOEFF_NORMED
        try:
            data = Settings.template_matching['method']
        except KeyError:
            print('[ERROR] Template matching method not found in config file. Default method is TM_CCOEFF_NORMED.')

        return data

    @staticmethod
    def get_threshold():
        data = 0
        try:
            data = Settings.template_matching['threshold']
        except KeyError:
            print('[ERROR] Template matching threshold not found in config file. Default threshold is 0.8.')

        return data


class Keys:
    @staticmethod
    def get_keys():
        data = []
        try:
            data = Settings.controls['keys']
        except KeyError:
            print('[ERROR] Keys enabled not found in config file. Default is disabled.')

        return data

    @staticmethod
    def get_mouse():
        data = []
        try:
            data = Settings.controls['mouse']
        except KeyError:
            print('[ERROR] Mouse enabled not found in config file. Default is disabled.')

        return data

    @staticmethod
    def get_key_mapping_size():
        from src.Utils.key_mapping import KeyMapping
        km = KeyMapping()
        return km.get_mapping_size()
