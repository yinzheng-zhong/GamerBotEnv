"""
This class will continuously capture screenshot from the Sensor. A maximum amount of screenshots can be stored in the
list. We some historical screenshot and a latest screenshot that has a action occurred.

"""
import collections
import queue as q
from multiprocessing import Process, Manager

import pyautogui
import threading
import time
import numpy as np
from mss import mss
import cv2
import src.Utils.other as other

from src.Helper.configs import NN, Capturing
import src.Utils.image as image_utils

FRAME_TIME_QUEUE_SIZE = 10


class Video:
    def __init__(self, ret_queue):
        self.frame_rate = Capturing.get_frame_rate()
        self.max_screenshots = NN.get_time_steps()
        self.resolution = Capturing.get_resolution()

        self.timestamps = collections.deque(maxlen=FRAME_TIME_QUEUE_SIZE)
        self.screenshot_list = ret_queue

        self.timestamps.extend(range(FRAME_TIME_QUEUE_SIZE))

        self.killed = False

    def capture_latest(self):
        while True:
            #print("Capturing latest")
            if self.killed:
                break

            if time.time() - self.timestamps[-1] < 1 / self.frame_rate:
                continue
            self.timestamps.append(time.time())  # log start time

            image = pyautogui.screenshot()

            image = np.array(image).astype(np.uint8)
            image = image_utils.convert_color_to_rgb(image)

            if all(self.resolution):
                image = image_utils.scale_image(image, self.resolution)

            self.put_data(image)

    def put_data(self, data):
        """
        Put data into the list and manage the size
        :param data:
        :return:
        """
        try:
            if self.screenshot_list.qsize() > 1:
                try:
                    self.screenshot_list.get_nowait()
                except q.Empty:
                    pass

            self.screenshot_list.put(data)
        except FileNotFoundError:
            print("Pipeline dead.")
            self.killed = True

    def run(self):
        zeros = np.zeros((500, 500, 3), dtype=np.uint8)

        frame_rate_thread = threading.Thread(target=other.print_frame_rate, args=(self.timestamps, 'Video capture'))
        frame_rate_thread.start()

        self.put_data(zeros)
        self.capture_latest()

        print("Capture process started")
