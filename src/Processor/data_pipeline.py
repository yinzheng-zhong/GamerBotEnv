"""
The code will be used to prepare the all necessary data for the agent to use.
"""
import collections
import threading
import time

import numpy as np

from src.Sensor.video import Video as vCap
from src.Sensor.audio import Audio as aCap
import queue as q
from multiprocessing import Process, Queue

from src.Utils import other
from src.Utils.key_mapping import KeyMapping
from src.Helper.configs import NN, Capturing, Agent
import src.Sensor.actions as act
import src.Helper.constance as const
from src.Processor.input_processing import Preprocessing
from src.Processor.reward_processing import RewardProcessing


FRAME_TIME_QUEUE_SIZE = 10


class DataPipeline:
    def __init__(self, agent_class):
        self.agent_class = agent_class
        self.batch_size = NN.get_batch_size()

        '''initialise the video capturing process'''
        self.video_queue = Queue(maxsize=1)
        self.reward_video_queue = Queue(maxsize=1)

        self.video_cap = vCap(self.video_queue, self.reward_video_queue, Capturing.get_frame_rate())
        self.video_process = Process(target=self.video_cap.run)

        '''initialise the reward processing process'''

        self.reward_queue = Queue(maxsize=1)
        self.reward_process_0 = RewardProcessing(self.reward_video_queue, self.reward_queue, const.PATH_TEMPLATES)
        self.reward_process_proc_0 = Process(target=self.reward_process_0.run)

        # self.reward_process_1 = RewardProcessing(self.reward_processing_queue, self.reward_queue, const.PATH_TEMPLATES)
        # self.reward_process_proc_1 = Process(target=self.reward_process_1.run)

        '''initialise audio capturing process'''
        #self.audio = Queue(maxsize=1)

        self.audio_cap = aCap()  # aCap(self.audio)
        self.audio_cap.run()

        #self.last_audio_buffer = None

        '''initialise the key monitoring process'''
        self.key_queue = Queue(maxsize=1)

        self.key_act = act.KeyMonitor(self.key_queue)
        self.key_listen_proc = Process(target=self.key_act.start_listening)

        '''instantiate the key mapping class'''

        self.key_mapping = KeyMapping()

        self.temp_data = Queue()  # for input processing. input processing will grab data  from here once available.
        # for agent. agent will grab data from here once available.
        self.processed_state_action_queue = Queue(maxsize=2)

        self.input_process = Preprocessing(self.temp_data, self.processed_state_action_queue)
        self.input_process_process = Process(target=self.input_process.run)

        self.agent = agent_class(self.processed_state_action_queue)
        self.agent_process = Process(target=self.agent.run)

        self.timestamps = collections.deque(maxlen=FRAME_TIME_QUEUE_SIZE)
        self.timestamps.extend(range(FRAME_TIME_QUEUE_SIZE))

        self.video_process.start()
        self.reward_process_proc_0.start()
        self.key_listen_proc.start()
        self.input_process_process.start()
        self.agent_process.start()

        self.last_screenshot = np.zeros((500, 500, 3), dtype=np.uint8)  # the init a screenshot

    def retrieve_screenshot(self):
        try:
            self.last_screenshot = self.video_queue.get_nowait()
        except q.Empty:
            pass

        return self.last_screenshot

    def retrieve_reward(self):
        try:
            return self.reward_queue.get_nowait()
        except q.Empty:
            pass

        return Agent.get_default_reward()  # default reward

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

    def check_procs(self):
        try:
            self.video_process.is_alive()
        except OSError:
            print('\033[93m\nReward process is dead.\033[0m')
            self.video_cap = vCap(self.video_queue, Capturing.get_frame_rate())
            self.video_process = Process(target=self.video_cap.run)

        try:
            self.reward_process_proc_0.is_alive()
        except OSError:
            print('\033[93m\nReward process is dead.\033[0m')
            self.reward_process_0 = RewardProcessing(self.reward_video_queue, self.reward_queue, const.PATH_TEMPLATES)
            self.reward_process_proc_0 = Process(target=self.reward_process_0.run)

        # try:
        #     self.reward_process_proc_1.is_alive()
        # except OSError:
        #     print('\033[93m\nReward process is dead.\033[0m')
        #     self.reward_process_1 = RewardProcessing(self.reward_processing_queue, self.reward_queue, const.PATH_TEMPLATES)
        #     self.reward_process_proc_1 = Process(target=self.reward_process_1.run)

        try:
            self.key_listen_proc.is_alive()
        except OSError:
            print('\033[93m\nKey listen process is dead.\033[0m')
            self.key_act = act.KeyMonitor(self.key_queue)
            self.key_listen_proc = Process(target=self.key_act.start_listening)

        if not self.input_process_process.is_alive():
            print('\033[93m\nInput process process is dead.\033[0m')
            self.input_process_process.start()

        try:
            self.input_process_process.is_alive()
        except OSError:
            print('\033[93m\nInput process process is dead.\033[0m')
            self.input_process = Preprocessing(self.temp_data, self.processed_state_action_queue)
            self.input_process_process = Process(target=self.input_process.run)

        if not self.agent_process.is_alive():
            print('\033[93m\nAgent process is dead.\033[0m')
            self.agent_process.start()

        try:
            self.agent_process.is_alive()
        except OSError:
            print('\033[93m\nAgent process is dead.\033[0m')
            self.agent = self.agent_class(self.processed_state_action_queue)
            self.agent_process = Process(target=self.agent.run)

    def start(self):
        """
        This is not a training batch. It is a batch that we use for speeding up the preprocessing.
        :return: {'x': [{'screenshot':, 'audio_l':, 'audio_r':}], 'y':[{'action':, 'cursor':}]}
        """

        '''start frame rate'''
        frame_rate_thread = threading.Thread(target=other.print_frame_rate, args=(self.timestamps, 'Collection'))
        frame_rate_thread.start()

        while True:
            self.check_procs()
            # collect data at some rate.
            if time.time() - self.timestamps[-1] < 1 / Capturing.get_frame_rate():
                continue

            self.timestamps.append(time.time())  # record start time

            key = self.retrieve_key_action()

            screenshot = self.retrieve_screenshot()
            reward = self.retrieve_reward()
            # cursor and audio are basically always available.
            audio_l, audio_r = self.retrieve_last_audio_buffer()

            state = {'screenshot': screenshot, 'audio_l': audio_l, 'audio_r': audio_r}

            self.temp_data.put({'state': state, 'action': key, 'reward': reward})
