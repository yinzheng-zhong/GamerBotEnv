import time

from pynput.keyboard import Listener as key_listener
from pynput.mouse import Listener as mouse_listener
import queue as q
from threading import Thread
from src.Helper.constance import MOUSE_MOVE_LEFT, MOUSE_MOVE_RIGHT, MOUSE_MOVE_UP, MOUSE_MOVE_DOWN, MOUSE_STOP_X, \
    MOUSE_STOP_Y, KEY_UNDEFINED

from multiprocessing import Process, Queue

from src.Helper.configs import Capturing


class KeyMonitor:
    def __init__(self, queue_key):
        """data_pipeline needs to be passed to the class"""
        self.queue_key = queue_key

        self.last_put_key = None
        self.frame_time = 1 / Capturing.get_frame_rate()
        self.holding_keys = []

        self.last_x, self.last_y, self.current_x, self.current_y = 0, 0, 0, 0
        self.last_x_time, self.last_y_time = 0, 0

        self.on_move_last_time = time.time()

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
            return KEY_UNDEFINED

        key_str = self.convert_to_string(key)
        if key_str not in self.holding_keys:
            self.put_data_important((key_str, True))

            self.last_put_key = key_str

            self.holding_keys.append(key_str)

    def remove_holding_keys(self, key_str):
        while key_str in self.holding_keys:
            self.holding_keys.remove(key_str)

    def on_hold(self):
        while True:
            time.sleep(self.frame_time)

            self.check_on_not_move()

            if len(self.holding_keys) <= 0:
                continue

            key_str = self.holding_keys[-1]
            self.put_data_not_important((key_str, True))

    def on_release(self, key):
        if key is None:
            return KEY_UNDEFINED

        key_str = self.convert_to_string(key)
        self.put_data_important((key_str, False))

        self.remove_holding_keys(key_str)

    def on_click(self, x, y, button, pressed):
        key_str = str(button)
        self.put_data_important((key_str, pressed))
        if pressed:
            self.holding_keys.append(key_str)
        else:
            self.remove_holding_keys(key_str)

    def convert_mouse_to_x_direction(self):
        if self.current_x > self.last_x + 10:
            self.last_x_time = time.time()
            return MOUSE_MOVE_RIGHT
        elif self.current_x < self.last_x + 10:
            self.last_x_time = time.time()
            return MOUSE_MOVE_LEFT

    def convert_mouse_to_y_direction(self):
        if self.current_y > self.last_y + 10:
            self.last_y_time = time.time()
            return MOUSE_MOVE_DOWN
        elif self.current_y < self.last_y + 10:
            self.last_y_time = time.time()
            return MOUSE_MOVE_UP

    def on_move(self, x, y):
        self.on_move_last_time = time.time()

        self.last_x, self.last_y = self.current_x, self.current_y
        self.current_x, self.current_y = x, y

        direction_x = self.convert_mouse_to_x_direction()
        direction_y = self.convert_mouse_to_y_direction()

        if direction_x and direction_x not in self.holding_keys:
            self.put_data_important((direction_x, True))
            self.holding_keys.append(direction_x)
        if direction_y and direction_y not in self.holding_keys:
            self.put_data_important((direction_y, True))
            self.holding_keys.append(direction_y)

    def check_on_not_move(self):
        time.sleep(self.frame_time)
        current_time = time.time()

        if current_time > self.last_x_time:
            if MOUSE_MOVE_RIGHT in self.holding_keys:
                self.put_data_important((MOUSE_STOP_X, True))  # here it is True because mouse action is not key
                self.remove_holding_keys(MOUSE_MOVE_RIGHT)
            if MOUSE_MOVE_LEFT in self.holding_keys:
                self.put_data_important((MOUSE_STOP_X, True))
                self.remove_holding_keys(MOUSE_MOVE_LEFT)

        if current_time > self.last_y_time:
            if MOUSE_MOVE_DOWN in self.holding_keys:
                self.put_data_important((MOUSE_STOP_Y, True))
                self.remove_holding_keys(MOUSE_MOVE_DOWN)
            if MOUSE_MOVE_UP in self.holding_keys:
                self.put_data_important((MOUSE_STOP_Y, True))
                self.remove_holding_keys(MOUSE_MOVE_UP)

    def start_listening(self):
        listener_key = key_listener(on_press=self.on_press, on_release=self.on_release)
        listener_mouse = mouse_listener(on_click=self.on_click)
        on_hold = Thread(target=self.on_hold)

        listener_key.start()
        listener_mouse.start()
        on_hold.start()
        print('Starting key listener')

        listener = mouse_listener(on_move=self.on_move)
        listener.start()
        print('Starting mouse listener')

        listener_key.join()
        listener_mouse.join()
        listener.join()
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
