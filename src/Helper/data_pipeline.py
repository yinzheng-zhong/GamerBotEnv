"""
The code will be used to prepare the all necessary data for the agent to use.
"""
import time

from src.Sensor.video import Capture as vCap
from src.Utils.image import extract_text as extract
from multiprocessing import Process


class DataPipeline:
    def __init__(self):
        self.video = vCap()

    def print_text(self):
        image = self.video.get_screenshot()
        text = extract(image)
        print(text)

"""test"""
if __name__ == "__main__":
    while True:
        capture = DataPipeline()
        capture.print_text()

        time.sleep(1)
        print("\n")
