"""
This tool takes screenshots from the game and saves them to var/screenshots. Then select screenshots
and crop them then place them under var/templates folder for template matching. The template matching
is used to detected the rewards.
"""

import os
import sys
from src.Sensor.video import Video
from multiprocessing import Process, Queue
import src.Utils.image as image_utils
import time
import src.Helper.constance as constance


if __name__ == "__main__":
    path = '../' + constance.PATH_SCREENSHOTS
    # check if var/screenshots directory exists
    if not os.path.exists(path):
        os.makedirs(path)

    cap_queue = Queue(maxsize=1)

    vcap = Video(cap_queue, frame_rate=1)
    vcap_process = Process(target=vcap.run)
    vcap_process.start()

    while True:
        time.sleep(0.1)
        if cap_queue.empty():
            continue
        else:
            frame = cap_queue.get()
            if frame is None:
                break
            else:
                filename = '{}{}.png'.format(path, time.time())
                print('Saving screenshot to {}'.format(filename))
                image_utils.save_image(frame, filename)
