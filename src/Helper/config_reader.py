from etc.settings import Settings
import cv2
import pyaudio


class NN:
    @staticmethod
    def get_time_steps():
        data = ''
        try:
            data = Settings.neural_network['time_steps']
        except KeyError:
            print('[ERROR] Neural network time steps not found in config file.')

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

    @staticmethod
    def get_audio_mixer_id():
        data = -1
        try:
            data = Settings.capturing['audio_stereo_mixer_device_id']
        except KeyError:
            print('[ERROR] Capturing audio mixer id not found in config file. Default audio mixer id is -1.')

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
    def get_keys_enabled():
        data = []
        try:
            data = Settings.keys['enabled']
        except KeyError:
            print('[ERROR] Keys enabled not found in config file. Default is disabled.')

        return data
