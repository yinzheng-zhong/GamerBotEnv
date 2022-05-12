import time

from pynput.keyboard import Listener as key_listener
from pynput.mouse import Listener as mouse_listener

from multiprocessing import Process, Queue


class KeyboardMonitor:
    def __init__(self, queue):
        """data_pipeline needs to be passed to the class"""
        self.queue_key = queue

    def on_press(self, key):
        if key is None:
            return

        try:
            '''alphanumeric key'''
            self.queue_key.put((key.char, True))
        except AttributeError:
            '''special key'''
            self.queue_key.put((str(key), True))

    def on_release(self, key):
        if key is None:
            return

        try:
            '''alphanumeric key'''
            self.queue_key.put((key.char, False))
        except AttributeError:
            '''special key'''
            self.queue_key.put((str(key), False))

    def start_listening(self):
        listener = key_listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        listener.join()


class MouseKeyMonitor:
    def __init__(self, queue):
        self.queue_key = queue

    def on_click(self, x, y, button, pressed):
        print('{0} at {1} with {2}'.format(
            'Mouse Pressed' if pressed else 'Released',
            (x, y), button))

        self.queue_key.put((x, y, str(button), pressed))

    def start_listening(self):
        listener = mouse_listener(on_click=self.on_click)
        listener.start()
        listener.join()


class MouseCursorMonitor:
    def __init__(self, queue):
        self.queue_key = queue
        self.last_time = 0

    def on_move(self, x, y):
        if self.queue_key.full():
            self.queue_key.get()

        if time.time() - self.last_time > 0.01:
            self.queue_key.put((x, y))
            self.last_time = time.time()

    def start_listening(self):
        listener = mouse_listener(on_move=self.on_move)
        listener.start()
        listener.join()
