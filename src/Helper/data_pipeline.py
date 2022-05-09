"""
The code will be used to prepare the all necessary data for the agent to use.
"""
import time

from src.Sensor.video import Capture as vCap
import src.Utils.image as img_util
from multiprocessing import Process, Manager, Queue
from src.Utils.buffer_queue import BufferQueue
from src.Helper.config_reader import NN
import src.Model.feature_mapping as fm


class DataPipeline:
    def __init__(self):
        manager = Manager()
        self.video_cap = vCap()
        self.video = manager.list()
        self.video_process = Process(target=self.video_cap.run, args=(self.video, ))
        self.video_process.start()

        input_x = []
        output_y = []

    def print_text(self):
        try:
            image = self.video[-1]
        except IndexError:
            print('waiting for image')
            return

        feature = img_util.load_image('target_destroyed.png')
        #match = img_util.feature_matching(image, feature)
        match = fm.check_template(image, feature)
        #text = extract(image)

        print(match)

        if match[0]:
            print('='*256)
            exit(0)

    def key_action(self):



"""test"""
if __name__ == "__main__":
    capture = DataPipeline()
    while True:
        capture.print_text()
        time.sleep(0.1)
        #print("\n")
