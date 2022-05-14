import time

from pynput.keyboard import Listener as key_listener
from pynput.mouse import Listener as mouse_listener
import queue as q

from multiprocessing import Process, Queue


class KeyMonitor:
    def __init__(self, queue):
        """data_pipeline needs to be passed to the class"""
        self.queue_key = queue

    def on_press(self, key):
        if key is None:
            return 'None'

        try:
            '''alphanumeric key'''
            self.queue_key.put((key.char, True))
        except AttributeError:
            '''special key'''
            self.queue_key.put((str(key), True))

    def on_release(self, key):
        if key is None:
            return 'None'

        try:
            '''alphanumeric key'''
            self.queue_key.put((key.char, False))
        except AttributeError:
            '''special key'''
            self.queue_key.put((str(key), False))

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

        listener_key.join()
        listener_mouse.join()


class MouseKeyMonitor:
    """
    Not used anymore
    """
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
        listener.join()
