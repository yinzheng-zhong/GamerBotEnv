import cv2
from pynput.keyboard import Key
import pyaudio
import src.Helper.constance as constance

"""
This is a sample settings file. rename it to settings.py before running the program.
"""


class Settings:
    capturing = {
        # Limit the frame rate. If your agent cannot catch up and the training queue is
        # piling up, you can try reducing this value. Normally the training queue should
        # have 0 or 1 item instead of 1 or 2.
        'video_frame_rate': 10,

        # (width, height) (0, 0) for original resolution keep below default.
        'video_resolution': (0, 0),
        'audio_format': pyaudio.paFloat32,
        'audio_sample_rate': 44100,
        'audio_length': 2,  # seconds
    }

    hardware = {
        # Your screen resolution. Must be the actual resolution.
        'screen_resolution': (2560, 1440),

        # the device ID for the stereo mixer. -1 for auto detection
        'audio_stereo_mixer_device_id': -1,

        # -1 for the rest of the GPUs except the first one
        'use_gpu_ids': -1,
    }

    """-------------------------------------------------------------------------------------------------"""
    template_matching = {
        'method': cv2.TM_CCOEFF_NORMED,
    }

    """-------------------------------------------------------------------------------------------------"""
    neural_network = {
        'time_steps': 5,
        'batch_size': 4,
        'screenshot_input_dim': (320, 180),
        'model': constance.NN_MODEL_LSTM,

        'training_queue_size': 1000,
    }

    """-------------------------------------------------------------------------------------------------"""
    agent = {
        # Same reward will not be given again within the gap. Some game displays the image for a while
        # in multiple frames. This settings prevents the agent from getting the same reward multiple times.
        'reward_time_gap': 5,

        # Agent controls the keyboard. If it is set to False. the controller will discard
        # the predicted actions
        'agent_control': True,

        'default_reward': 0
    }

    """-------------------------------------------------------------------------------------------------"""
    keys = {
        'enabled': ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p",
            "a", "s", "d", "f", "g", "h", "j", "k", "l", "z", "x", "c", "v", "b", "n", "m", ",", ".",
            "Key.space", "Key.shift", "Key.esc", "Key.enter", "Key.backspace", "Key.tab",
            "Key.caps_lock", "Key.left", "Key.up", "Key.right", "Key.down", "Key.alt",
            "Key.f1", "Key.f2", "Key.f3", "Key.f4", "Key.f5", "1", "2", "3", "4", "5",
            "6", "0", "Key.ctrl_l"] + ["Button.left", "Button.right"]
    }
