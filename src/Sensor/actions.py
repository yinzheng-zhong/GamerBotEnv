import time

from pynput.keyboard import Listener as key_listener
from pynput.mouse import Listener as mouse_listener
import queue as q
from threading import Thread

from multiprocessing import Process, Queue

from src.Helper.configs import Capturing


class KeyMonitor:
    def __init__(self, queue):
        """data_pipeline needs to be passed to the class"""
        self.queue_key = queue

        self.last_put_key = None
        self.frame_time = 1 / Capturing.get_frame_rate()
        self.holding_keys = []

    def put_data_not_important(self, data):
        """
        Put data into the list and manage the size
        :param data:
        :return:
        """
        try:
            try:
                self.queue_key.put_nowait(data)
            except q.Full:
                pass

        except FileNotFoundError:
            print("Pipeline dead.")

    def put_data_important(self, data):
        """
        Put data into the list and manage the size
        :param data:
        :return:
        """
        try:
            try:
                self.queue_key.put_nowait(data)
            except q.Full:
                try:
                    self.queue_key.get_nowait()
                except q.Empty:
                    pass

                self.queue_key.put(data)

        except FileNotFoundError:
            print("Pipeline dead.")

    def convert_to_string(self, key):
        try:
            '''alphanumeric key'''
            return str.lower(key.char)
        except AttributeError:
            '''special key'''
            return str(key)

    def on_press(self, key):
        if key is None:
            return 'None'

        key_str = self.convert_to_string(key)
        if key_str not in self.holding_keys:
            self.put_data_important((key_str, True))

            self.last_put_key = key_str

            self.holding_keys.append(key_str)

    def on_hold(self):
        while True:
            time.sleep(self.frame_time)

            if len(self.holding_keys) <= 0:
                continue

            key_str = self.holding_keys[-1]
            self.put_data_not_important((key_str, True))

    def on_release(self, key):
        if key is None:
            return 'None'

        key_str = self.convert_to_string(key)
        self.put_data_important((key_str, False))

        if key_str in self.holding_keys:
            self.holding_keys.remove(key_str)

    def on_click(self, x, y, button, pressed):
        key_str = str(button)
        self.put_data_important((key_str, pressed))
        if pressed:
            self.holding_keys.append(key_str)
        else:
            self.holding_keys.remove(key_str)

    def start_listening(self):
        listener_key = key_listener(on_press=self.on_press, on_release=self.on_release)
        listener_mouse = mouse_listener(on_click=self.on_click)
        on_hold = Thread(target=self.on_hold)

        listener_key.start()
        listener_mouse.start()
        on_hold.start()
        print('Starting key listener')

        listener_key.join()
        listener_mouse.join()
        on_hold.join()


class MouseCursorMonitor:
    def __init__(self, queue):
        self.queue_key = queue
        self.last_time = 0

    def on_move(self, x, y):
        if time.time() - self.last_time > 0.01:
            self.last_time = time.time()
            if self.queue_key.full():
                try:
                    self.queue_key.get_nowait()
                except q.Empty:
                    pass

            self.queue_key.put((x, y))

        # # reduce the sample rate and limit to 100 samples per second
        # if time.time() - self.last_time > 0.01:
        #     self.queue_key.put((x, y))
        #     self.last_time = time.time()

    def start_listening(self):
        listener = mouse_listener(on_move=self.on_move)
        listener.start()
        print('Starting mouse listener')
        listener.join()
