from etc.settings import Settings


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
        data = (512, 512)

        try:
            data = Settings.capturing['resolution']
        except KeyError:
            print('[ERROR] Capturing resolution not found in config file. Default resolution is 512x512.')

        return data
