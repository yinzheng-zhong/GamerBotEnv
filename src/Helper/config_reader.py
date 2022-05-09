from etc.settings import Settings
import cv2


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
            data = Settings.capturing['frame_rate']
        except KeyError:
            print('[ERROR] Capturing frame rate not found in config file.')

        return data

    @staticmethod
    def get_resolution():
        data = (0, 0)

        try:
            data = Settings.capturing['resolution']
        except KeyError:
            print('[ERROR] Capturing resolution not found in config file. Default resolution is 512x512.')

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
