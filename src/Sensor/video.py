"""
This class will continuously capture screenshot from the Sensor. A maximum amount of screenshots can be stored in the
list. We some historical screenshot and a latest screenshot that has a action occurred.

"""
import collections
from multiprocessing import Process

import pyautogui
import threading
import time
import numpy as np

from src.Helper.config_reader import NN, Capturing
import src.Utils.image as image_utils

FRAME_TIME_QUEUE_SIZE = 10


class Capture:
    def __init__(self):
        self.frame_rate = Capturing.get_frame_rate()
        self.max_screenshots = NN.get_time_steps()
        self.resolution = Capturing.get_resolution()

        self.screenshot_list = collections.deque(maxlen=self.max_screenshots)
        self.timestamps = collections.deque(maxlen=FRAME_TIME_QUEUE_SIZE)

        # initialize the deque
        self.screenshot_list.extend(
            [np.zeros((self.resolution[0], self.resolution[1], 3), dtype=np.uint8)] * self.max_screenshots
        )
        self.timestamps.extend(range(FRAME_TIME_QUEUE_SIZE))

        self.run()

    def capture_latest(self):
        while True:
            if time.time() - self.timestamps[-1] < 1 / self.frame_rate:
                time.sleep(0.01)
                continue

            image = pyautogui.screenshot()
            image = np.array(image).astype(np.uint8)

            if any(self.resolution):
                image = image_utils.scale_image(image, self.resolution)
            else:
                image = image

            # if time.time() - self.last_captured > 1 / self.frame_rate:
            #     self.screenshot_list.append(image)
            #     self.last_captured = time.time()

            self.screenshot_list.append(image)
            self.timestamps.append(time.time())

    def print_frame_rate(self):
        while True:
            mean_time = (self.timestamps[-1] - self.timestamps[0]) / (FRAME_TIME_QUEUE_SIZE - 1)
            print("Actual frame rate: {}".format(1 / mean_time))
            time.sleep(1)

    def run(self):
        capture_process = threading.Thread(target=self.capture_latest)
        capture_process.start()

        frame_rate_process = threading.Thread(target=self.print_frame_rate)
        frame_rate_process.start()

        print("Capture thread started")
        #capture_process.join()

    def get_screenshot_series(self):
        """
        Returns a list of screenshots for time series prediction
        """
        return list(self.screenshot_list)

    def get_screenshot(self):
        """
        Returns the latest screenshot for single prediction
        """
        return list(self.screenshot_list)[-1]


# """test"""
# if __name__ == "__main__":
#     capture = Capture()
#     capture.run()
#     print("done")
