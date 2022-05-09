"""
The code will be used to prepare the all necessary data for the agent to use.
"""
import time

from src.Sensor.video import Capture as vCap
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
        manager = Manager()

        '''initialise the video capturing process'''
        self.video = Queue(maxsize=1)

        self.video_cap = vCap(self.video)
        self.video_process = Process(target=self.video_cap.run)
        self.video_process.start()

        self.last_screenshot = None

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
        self.mouse_cursor_queue = Queue()

        self.mouse_cursor = act.MouseCursorMonitor(self.mouse_cursor_queue)
        self.mouse_cursor_process = Process(target=self.mouse_cursor.start_listening)
        self.mouse_cursor_process.start()

        self.last_mouse_pos = (0, 0)

        '''instantiate the key mapping class'''

        self.key_mapping = KeyMapping()

        dataset = {'x': [], 'y': []}

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
        if self.video.empty():
            screenshot = self.last_screenshot
        else:
            screenshot = self.video.get()

        self.last_screenshot = screenshot
        return screenshot

    def retrieve_key_action(self):
        if self.key_queue.empty():
            return

        key, pressed = self.key_queue.get()
        print(key, pressed)

        # get_rid of extra code
        #key = str.split(key, ' ')[0]

        if not pressed:
            key += const.KEY_RELEASE_SUFFIX

        output_action_vector = self.key_mapping.get_on_hot_mapping(key)
        print(output_action_vector)

        return output_action_vector

    def retrieve_mouse_key_action(self):
        if self.mouse_key_queue.empty():
            return

        x, y, key, pressed = self.mouse_key_queue.get()

        if not pressed:
            key += const.KEY_RELEASE_SUFFIX

        output_action_vector = self.key_mapping.get_on_hot_mapping(key)
        print(output_action_vector)

        return output_action_vector

    def retrieve_mouse_cursor_action(self):
        if self.mouse_cursor_queue.empty():
            x, y = self.last_mouse_pos
        else:
            x, y = self.mouse_cursor_queue.get()

        self.last_mouse_pos = (x, y)

        return x, y

    def make_batch(self):
        """
        TODO: Continue here. every action will pair with a image and the last cursor position. if no action is available
        TODO: , use the last image with default actions and last cursor position.
        :return:
        """

"""test"""
if __name__ == "__main__":
    dp = DataPipeline()
    while True:
        #dp.print_text()
        dp.retrieve_key_action()
        dp.retrieve_mouse_key_action()
        time.sleep(0.001)
        #print("\n")
