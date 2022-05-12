"""
The code will be used to prepare the all necessary data for the agent to use.
"""
import time

from src.Sensor.video import Video as vCap
from src.Sensor.audio import Audio as aCap
import src.Utils.image as img_util
import multiprocessing
from multiprocessing import Process, Manager, Queue
from src.Utils.key_mapping import KeyMapping
from src.Helper.config_reader import NN
import src.Model.feature_mapping as fm
import src.Sensor.actions as act
import src.Helper.constance as const


class DataPipeline:
    def __init__(self):
        self.batch_size = NN.get_batch_size()

        '''initialise the video capturing process'''
        self.video = Queue(maxsize=1)

        self.video_cap = vCap(self.video)
        self.video_process = Process(target=self.video_cap.run)
        self.video_process.start()

        self.last_screenshot = None  # the avoid the queue starvation

        '''initialise audio capturing process'''
        #self.audio = Queue(maxsize=1)

        self.audio_cap = aCap()  # aCap(self.audio)

        #self.last_audio_buffer = None

        '''initialise the key monitoring process'''
        self.key_queue = Queue()

        self.keyboard = act.KeyboardMonitor(self.key_queue)
        self.keyboard_process = Process(target=self.keyboard.start_listening)
        self.keyboard_process.start()

        '''initialise the mouse key monitoring process'''
        self.mouse_key_queue = Queue()

        self.mouse_key = act.MouseKeyMonitor(self.mouse_key_queue)
        self.mouse_process = Process(target=self.mouse_key.start_listening)
        self.mouse_process.start()

        '''initialise the mouse cursor monitoring process'''
        self.mouse_cursor_queue = Queue(maxsize=1)

        self.mouse_cursor = act.MouseCursorMonitor(self.mouse_cursor_queue)
        self.mouse_cursor_process = Process(target=self.mouse_cursor.start_listening)
        self.mouse_cursor_process.start()

        self.last_mouse_pos = (0, 0)

        '''instantiate the key mapping class'''

        self.key_mapping = KeyMapping()

        self.temp_data = Queue()  # for input processing. input processing will grab data  from here once available.

        self.dataset = Queue()  # for agent. agent will grab data from here once available.

    # def print_text(self):
    #     try:
    #         image = self.video.get()
    #     except IndexError:
    #         print('waiting for image')
    #         return
    #
    #     feature = img_util.load_image('target_destroyed.png')
    #     match = fm.check_single_template(image, feature)
    #
    #     print(match)
    #
    #     if match[0]:
    #         print('='*256)
    #         exit(0)

    def retrieve_screenshot(self):
        try:
            screenshot = self.video.get(block=True, timeout=0.01)
            self.last_screenshot = screenshot
            return screenshot, True  # is_new
        except Exception as e:
            print(e)
            return self.last_screenshot, False

    def retrieve_last_audio_buffer(self):
        return self.audio_cap.get_audio()

    def retrieve_key_action(self):
        if self.key_queue.empty():
            return None

        key, pressed = self.key_queue.get()
        print('Key: ' + str(key), pressed)

        # get_rid of extra code
        #key = str.split(key, ' ')[0]

        if not pressed:
            key += const.KEY_RELEASE_SUFFIX

        # output_action_vector = self.key_mapping.get_on_hot_mapping(key)
        # print(output_action_vector)

        return key

    def retrieve_mouse_key_action(self):
        if self.mouse_key_queue.empty():
            return None

        x, y, key, pressed = self.mouse_key_queue.get()
        print('Key: ' + str(key), pressed)

        if not pressed:
            key += const.KEY_RELEASE_SUFFIX

        # output_action_vector = self.key_mapping.get_on_hot_mapping(key)
        # print(output_action_vector)
        self.last_mouse_pos = (x, y)

        return key

    def retrieve_mouse_cursor_pos(self):
        try:
            x, y = self.mouse_cursor_queue.get(block=True, timeout=0.01)
            self.last_mouse_pos = (x, y)
            return x, y
        except Exception as e:
            print(e)
            return self.last_mouse_pos

    def make_batch(self):
        """
        :return: {'x': [{'screenshot':, 'audio_l':, 'audio_r':}], 'y':[{'action':, 'cursor':}]}
        """
        data = {'x': [], 'y': []}
        counter = 0

        while counter < self.batch_size:
            # only record data when getting new screenshot or actions
            # screenshot is special because even if it is not available, it will still be used to match the new actions.
            screenshot, is_new_sct = self.retrieve_screenshot()

            keyboard = self.retrieve_key_action()
            mouse_key = self.retrieve_mouse_key_action()

            # cursor and audio are basically always available.
            audio_l, audio_r = self.retrieve_last_audio_buffer()
            cursor = self.retrieve_mouse_cursor_pos()

            if keyboard is not None and mouse_key is not None:
                data['x'].append({'screenshot': screenshot, 'audio_l': audio_l, 'audio_r': audio_r})
                data['y'].append({'action': keyboard, 'cursor': cursor})

                counter += 1
                if counter >= self.batch_size - 1:
                    break

                data['x'].append({'screenshot': screenshot, 'audio_l': audio_l, 'audio_r': audio_r})
                data['y'].append({'action': mouse_key, 'cursor': cursor})

                counter += 1
            elif keyboard is not None and mouse_key is None:
                data['x'].append({'screenshot': screenshot, 'audio_l': audio_l, 'audio_r': audio_r})
                data['y'].append({'action': keyboard, 'cursor': cursor})

                counter += 1
            elif keyboard is None and mouse_key is not None:
                data['x'].append({'screenshot': screenshot, 'audio_l': audio_l, 'audio_r': audio_r})
                data['y'].append({'action': mouse_key, 'cursor': cursor})

                counter += 1

            elif is_new_sct:
                data['x'].append({'screenshot': screenshot, 'audio_l': audio_l, 'audio_r': audio_r})
                data['y'].append({'action': None, 'cursor': cursor})

                counter += 1

        self.temp_data.put(data)


"""test"""
if __name__ == "__main__":
    dp = DataPipeline()
    while True:
        #dp.print_text()
        dp.make_batch()
        time.sleep(0.001)
        #print("\n")
