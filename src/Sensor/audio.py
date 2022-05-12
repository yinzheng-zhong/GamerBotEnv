"""
We collect audio in real-time and keep them in a queue for certain amount of time, say 3s, so we can use it whenever
needed. Can not put the audio_interface in a new process for now. Will maybe look into this in the future.
"""
import collections

import pyaudio
from src.Helper.config_reader import Capturing, Hardware
import numpy as np
import audioop
import time
from threading import Thread


class Audio:
    def __init__(self):
        #self.audio = ret_queue

        self.audio_format = Capturing.get_audio_format()
        self.audio_sample_rate = Capturing.get_audio_sample_rate()
        self.audio_length = Capturing.get_audio_length()

        self.audio_interface = pyaudio.PyAudio()

        self.mixer = self.get_audio_mixer_id()

        self.chunk = int(self.audio_sample_rate * self.audio_length / 60)

        self.audio_buffer_queue = collections.deque(maxlen=60)

        _ = [self.audio_buffer_queue.append(np.zeros((self.chunk * 2, ))) for _ in range(60)]

        self.stream = self.audio_interface.open(
            format=self.audio_format,
            channels=2,
            rate=self.audio_sample_rate,
            frames_per_buffer=self.chunk,
            input=True,
            input_device_index=self.mixer,
        )

        self.killed = False

    def get_audio_mixer_id(self):
        if Hardware.get_audio_mixer_id() != -1:
            return Hardware.get_audio_mixer_id()

        for i in range(self.audio_interface.get_device_count()):
            dev = self.audio_interface.get_device_info_by_index(i)
            if 'Stereo Mix' in dev['name'] and dev['hostApi'] == 0:
                dev_index = dev['index']
                print('Stereo Mix Dev Index:', dev_index)
                return dev_index

    def cap_frame(self):
        while True:
            if self.audio_format == pyaudio.paInt16:
                audio_format = np.int16
            elif self.audio_format == pyaudio.paInt32:
                audio_format = np.int32
            elif self.audio_format == pyaudio.paFloat32:
                audio_format = np.float32
            elif self.audio_format == pyaudio.paInt8:
                audio_format = np.int8
            else:
                raise ValueError("Unsupported audio format")

            stream = self.stream.read(self.chunk)

            self.audio_buffer_queue.append(np.frombuffer(stream, dtype=audio_format))

    def get_audio(self):
        flattened = np.concatenate(self.audio_buffer_queue)
        start = time.time()

        separate_channel = np.reshape(flattened, (self.audio_sample_rate * self.audio_length, 2))

        #self.put_data((separate_channel[:, 0], separate_channel[:, 1]))
        return separate_channel[:, 0], separate_channel[:, 1]

    # def put_data(self, data):
    #     try:
    #         if self.audio.full():
    #             self.audio.get()
    #
    #         self.audio.put(data)
    #     except FileNotFoundError:
    #         print("Pipeline dead.")
    #         self.killed = True
    #
    def run(self):
        cap_thread = Thread(target=self.cap_frame)
        cap_thread.start()
