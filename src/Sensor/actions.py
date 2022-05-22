import time

from pynput.keyboard import Listener as key_listener
from pynput.mouse import Listener as mouse_listener
import queue as q

from multiprocessing import Process, Queue


class KeyMonitor:
    def __init__(self, queue):
        """data_pipeline needs to be passed to the class"""
        self.queue_key = queue

        self.last_put_key = None
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
            return key.char
        except AttributeError:
            '''special key'''
            return str(key)

    def on_press(self, key):
        if key is None:
            return 'None'

        key_str = self.convert_to_string(key)
        if key in self.holding_keys and self.last_put_key == key:
            self.put_data_not_important((key_str, True))  # will just try to put the key
        else:
            self.put_data_important((key_str, True))  # must wait and put important key

        self.last_put_key = key
        self.holding_keys.append(key)

    def on_release(self, key):
        if key is None:
            return 'None'

        key_str = self.convert_to_string(key)
        self.put_data_important((key_str, False))

        if key in self.holding_keys:
            self.holding_keys.remove(key)

    def on_click(self, x, y, button, pressed):
        print('{0} at {1} with {2}'.format(
            'Mouse Pressed' if pressed else 'Released',
            (x, y), button))

        self.queue_key.put((str(button), pressed))

    def start_listening(self):
        listener_key = key_listener(on_press=self.on_press, on_release=self.on_release)
        listener_mouse = mouse_listener(on_click=self.on_click)

        listener_key.start()
        listener_mouse.start()
        print('Starting key listener')

        listener_key.join()
        listener_mouse.join()


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
