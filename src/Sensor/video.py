"""
This class will continuously capture screenshot from the Sensor. A maximum amount of screenshots can be stored in the
list. We some historical screenshot and a latest screenshot that has a action occurred.

"""
import collections
from multiprocessing import Process

import pyautogui
import multiprocessing
import time
import numpy as np

from src.Helper.config_reader import NN, Capturing
import src.Utils.image as image_utils


class Capture:
    def __init__(self):
        self.frame_rate = Capturing.get_frame_rate()
        self.max_screenshots = NN.get_time_steps()
        self.resolution = Capturing.get_resolution()

        self.screenshot_list = collections.deque(maxlen=self.max_screenshots)

        # initialize the deque
        self.screenshot_list.extend(
            [np.zeros((self.resolution[0], self.resolution[1], 3))] * self.max_screenshots
        )

        self.run()

    def capture_latest(self):
        image = pyautogui.screenshot()
        image = np.array(image).astype(np.uint8)

        if any(self.resolution):
            return image_utils.scale_image(image, self.resolution)
        else:
            return image

    def capture_historical(self):
        while True:
            image = self.capture_latest()
            self.screenshot_list.append(image)

            time.sleep(1 / self.frame_rate)

    def run(self):
        #self.capture()

        capture_process = Process(target=self.capture_historical)
        capture_process.start()

        print("Capture process started")
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
