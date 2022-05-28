from multiprocessing import Process, Pool
import src.Helper.constance as constance
import src.Utils.image as image_utils
import src.Utils.feature_matching as feature_matching_utils
import os
import time
import queue as q
from src.Processor.timed_list import TimedList
from src.Helper.configs import Agent as AgentConfig
import numpy as np


class RewardProcessing:
    """
    The class use template matching to identify the reward. The reward will be put into a queue.
    """

    def __init__(self, image_queue, reward_queue, template_dir, detailed_print=False):
        """
        Initialize the class.
        :param image_queue: Queue
        :param reward_queue: Queue
        """

        self._image_queue = image_queue
        self._reward_queue = reward_queue
        self.testing = detailed_print

        self.template_folder = template_dir

        self._last_screenshot = None

        self.templates = {}
        self._load_templates()

        self.timed_list = TimedList(AgentConfig.get_reward_time_gap())
        self.default_reward = AgentConfig.get_default_reward()

    def put_data(self, data):
        """
        Put data into the list and manage the size
        :param data:
        :return:
        """# != 0 makes sure the reward won't be missed.
        try:
            self._reward_queue.put_nowait(data)
        except q.Full:
            pass

        self._reward_queue.put(data)

    def _load_templates(self):
        try:
            sub_folders = os.listdir(self.template_folder)
        except FileNotFoundError:
            print("Templates folder not found. Please read the README.md file.")
            return

        for sub_folder in sub_folders:
            sub_folder_dir = self.template_folder + sub_folder + '/'

            self.templates[sub_folder] = []
            for file in os.listdir(sub_folder_dir):
                self.templates[sub_folder].append(image_utils.load_image(sub_folder_dir + file))

    def _check_reward(self, pool):
        start = time.time()
        self._last_screenshot = self._image_queue.get()

        list_property = map(lambda x: x.split('+'), self.templates.keys())
        names, thresholds, rewards = zip(*list_property)

        zipped = zip(
            names,
            [self._last_screenshot] * len(names),
            self.templates.values(),
            [self.testing] * len(self.templates.keys())
        )

        result_list = list(pool.starmap(self._check_single_reward, zipped))

        reward_totals = 0
        for i, n in enumerate(names):
            result = result_list[i]
            reward = int(rewards[i])
            threshold = float(thresholds[i])
            if n not in self.timed_list and result > threshold:
                print(f"\033[93m\nGiving {reward} reward for {n}.\033[0m")
                self.timed_list.add(n, time.time())
                reward_totals += reward

        if self.testing:
            print('\n' + '=' * 20, f"Total time: {time.time() - start}", '=' * 20)

        if reward_totals == 0:
            reward_totals = self.default_reward

        return reward_totals

    @staticmethod
    def _check_single_reward(name, screenshot_map_arg, template_images, testing=False):
        start = time.time()

        screenshot_map_arg = [screenshot_map_arg] * len(template_images)

        # return pool.starmap(feature_matching_utils.template_matching, zip(screenshot_map_arg, template_images))
        results = list(map(feature_matching_utils.template_matching, screenshot_map_arg, template_images))
        if testing:
            print(f"\n{name} has the highest matching of {max(results)}.")
            print(f"{name} check time: {time.time() - start}")

        return max(results)

    def run(self):
        pool = Pool(processes=2)
        print("Starting reward checker.")

        while True:
            print("Checking reward...")
            total_reward = self._check_reward(pool)
            print(f"Total reward: {total_reward}")

            if total_reward > 0:
                print(f"\033[93m\nTotal reward: {total_reward}\033[0m")
                self.put_data(total_reward)
                print("Reward given.")
