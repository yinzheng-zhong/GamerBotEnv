import time

from src.Utils.key_mapping import KeyMapping
import src.Helper.constance as constance
from src.Helper.configs import Capturing, Hardware
import pyautogui
from threading import Thread


class Controller:
    def __init__(self, key_queue):
        self.key_queue = key_queue

        self.key_mapping = KeyMapping()
        self.mouse_x_state = constance.MOUSE_STOP_X
        self.mouse_y_state = constance.MOUSE_STOP_Y

        self._frame_rate = Capturing.get_frame_rate()
        self._frame_time = 1 / self._frame_rate
        self._screen_resolution = Hardware.get_screen_res()
        self.executed_keys = []

        self.mouse_move_step = 10

    def get_key_string(self):
        one_hot = self.key_queue.get()
        return self.key_mapping.get_key_from_on_hot_mapping(one_hot)

    def decode_key(self, key):
        if constance.MOUSE_KEY_PREFIX not in key and 'Button.' not in key:
            if key == 'idle':
                self.executed_keys = []
            elif constance.KEY_RELEASE_SUFFIX in key:
                key_no_suffix = key.replace(constance.KEY_RELEASE_SUFFIX, '')
                if key not in self.executed_keys:
                    if key_no_suffix in self.executed_keys:
                        self.executed_keys.remove(key_no_suffix)
                    self.executed_keys.append(key)
                    key_no_suffix = self.key_mapping.convert_pynput_to_pyautogui(key_no_suffix)
                    pyautogui.keyUp(key_no_suffix)
            elif key not in self.executed_keys:
                if key + constance.KEY_RELEASE_SUFFIX in self.executed_keys:
                    self.executed_keys.remove(key + constance.KEY_RELEASE_SUFFIX)
                self.executed_keys.append(key)
                key = self.key_mapping.convert_pynput_to_pyautogui(key)
                pyautogui.keyDown(key)

        elif 'Button.' in key:
            mouse_button = key.replace('Button.', '')
            if mouse_button == 'left':
                pyautogui.mouseDown(button='left')
            elif mouse_button == 'right':
                pyautogui.mouseDown(button='right')
            elif mouse_button == 'left' + constance.KEY_RELEASE_SUFFIX:
                pyautogui.mouseUp(button='left')
            elif mouse_button == 'right' + constance.KEY_RELEASE_SUFFIX:
                pyautogui.mouseUp(button='right')

        else:
            if key in (constance.MOUSE_MOVE_LEFT, constance.MOUSE_MOVE_RIGHT, constance.MOUSE_STOP_X):
                self.mouse_x_state = key
            elif key in (constance.MOUSE_MOVE_UP, constance.MOUSE_MOVE_DOWN, constance.MOUSE_STOP_Y):
                self.mouse_y_state = key

    def move_mouse(self):
        while True:
            if self.mouse_x_state == constance.MOUSE_MOVE_LEFT:
                pyautogui.moveRel(-self.mouse_move_step, 0, duration=0.01)
            elif self.mouse_x_state == constance.MOUSE_MOVE_RIGHT:
                pyautogui.moveRel(self.mouse_move_step, 0, duration=0.01)

            if self.mouse_y_state == constance.MOUSE_MOVE_UP:
                pyautogui.moveRel(0, -self.mouse_move_step, duration=0.01)
            elif self.mouse_y_state == constance.MOUSE_MOVE_DOWN:
                pyautogui.moveRel(0, self.mouse_move_step, duration=0.01)

            time.sleep(0.01)

    def run(self):
        mouse_thread = Thread(target=self.move_mouse)
        mouse_thread.start()

        while True:
            key = self.get_key_string()
            self.decode_key(key)

