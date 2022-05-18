"""
The code will be used to prepare the all necessary data for the agent to use.
"""
import collections
import threading
import time

from src.Sensor.video import Video as vCap
from src.Sensor.audio import Audio as aCap
import queue as q
from multiprocessing import Process, Queue

from src.Utils import other
from src.Utils.key_mapping import KeyMapping
from src.Helper.configs import NN, Capturing
import src.Sensor.actions as act
import src.Helper.constance as const
from src.Processor.input_processing import Preprocessing

FRAME_TIME_QUEUE_SIZE = 10


class DataPipeline:
    def __init__(self, agent_class):
        self.batch_size = NN.get_batch_size()

        '''initialise the video capturing process'''
        self.video = Queue(maxsize=1)

        self.video_cap = vCap(self.video, Capturing.get_frame_rate())
        self.video_process = Process(target=self.video_cap.run)

        '''initialise audio capturing process'''
        #self.audio = Queue(maxsize=1)

        self.audio_cap = aCap()  # aCap(self.audio)
        self.audio_cap.run()

        #self.last_audio_buffer = None

        '''initialise the key monitoring process'''
        self.key_queue = Queue()

        self.key_act = act.KeyMonitor(self.key_queue)
        self.key_listen_proc = Process(target=self.key_act.start_listening)

        '''initialise the mouse cursor monitoring process'''
        self.mouse_cursor_queue = Queue(maxsize=1)

        self.mouse_cursor = act.MouseCursorMonitor(self.mouse_cursor_queue)
        self.mouse_cursor_process = Process(target=self.mouse_cursor.start_listening)

        self.last_mouse_pos = (0, 0)

        '''instantiate the key mapping class'''

        self.key_mapping = KeyMapping()

        self.temp_data = Queue()  # for input processing. input processing will grab data  from here once available.
        # for agent. agent will grab data from here once available.
        self.training_queue = Queue(maxsize=NN.get_training_queue_size())

        self.input_process = Preprocessing(self.temp_data, self.training_queue)
        self.input_process_process = Process(target=self.input_process.run)

        self.agent = agent_class(self.training_queue)
        self.agent_process = Process(target=self.agent.run)

        self.timestamps = collections.deque(maxlen=FRAME_TIME_QUEUE_SIZE)
        self.timestamps.extend(range(FRAME_TIME_QUEUE_SIZE))

        self.video_process.start()
        self.key_listen_proc.start()
        self.mouse_cursor_process.start()
        self.input_process_process.start()
        self.agent_process.start()

        self.last_screenshot = self.video.get()  # the init a screenshot

    def retrieve_screenshot(self):
        try:
            self.last_screenshot = self.video.get_nowait()
            return self.last_screenshot
        except q.Empty:
            return self.last_screenshot

    def retrieve_last_audio_buffer(self):
        return self.audio_cap.get_audio()

    def retrieve_key_action(self):
        try:
            key, pressed = self.key_queue.get_nowait()
            print('Key: ' + str(key), pressed)

            if not pressed:
                key += const.KEY_RELEASE_SUFFIX

            return key

        except (TypeError, q.Empty):
            return None

    def retrieve_mouse_cursor_pos(self):
        try:
            self.last_mouse_pos = self.mouse_cursor_queue.get_nowait()
            return self.last_mouse_pos
        except q.Empty:
            return self.last_mouse_pos

    def start(self):
        """
        This is not a training batch. It is a batch that we use for speeding up the preprocessing.
        :return: {'x': [{'screenshot':, 'audio_l':, 'audio_r':}], 'y':[{'action':, 'cursor':}]}
        """
        counter = 0

        '''start frame rate'''
        frame_rate_thread = threading.Thread(target=other.print_frame_rate, args=(self.timestamps, 'Collection'))
        frame_rate_thread.start()

        while counter < self.batch_size:
            # collect data at some rate.
            if time.time() - self.timestamps[-1] < 1 / Capturing.get_frame_rate():
                continue

            self.timestamps.append(time.time())  # record start time

            screenshot = self.retrieve_screenshot()

            key = self.retrieve_key_action()

            # cursor and audio are basically always available.
            audio_l, audio_r = self.retrieve_last_audio_buffer()
            cursor = self.retrieve_mouse_cursor_pos()

            x = {'screenshot': screenshot, 'audio_l': audio_l, 'audio_r': audio_r}
            y = {'action': key, 'cursor': cursor}

            self.temp_data.put({'x': x, 'y': y})
