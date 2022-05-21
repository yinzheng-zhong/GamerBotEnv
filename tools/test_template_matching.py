import time
from multiprocessing import Process, Queue

from src.Sensor.video import Video
from src.Processor.reward_processing import RewardProcessing
import src.Helper.constance as constance

"""
After setting up template, capture some screenshots again. Open the screenshots and display 
the screenshot in full screen. Test with this script for a few seconds and terminate it to see if the template is found.
You can also test while running the game but you may not able to catch it, so you need to
set detailed_print to Flase.
"""


if __name__ == '__main__':
    cap_queue = Queue(maxsize=1)
    reward_queue = Queue(maxsize=1)

    vcap = Video(cap_queue, frame_rate=10)
    vcap_process = Process(target=vcap.run)
    vcap_process.start()

    reward = RewardProcessing(cap_queue, reward_queue, '../' + constance.PATH_TEMPLATES, detailed_print=False)
    reward_process = Process(target=reward.run)
    reward_process.start()

    while True:
        time.sleep(0.01)
        if reward_queue.empty():
            continue
        else:
            cap_queue.get()
