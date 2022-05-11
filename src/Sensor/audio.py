"""
We collect audio in real-time and keep them in a queue for certain amount of time, say 3s, so we can use it whenever
needed. I guess the pyaudio uses multithreading?
"""
import pyaudio
from src.Helper.config_reader import Capturing
import numpy as np
import audioop
import time


class Audio:
    def __init__(self, ret_queue):
        self.audio = ret_queue

        self.audio_format = Capturing.get_audio_format()
        self.audio_sample_rate = Capturing.get_audio_sample_rate()
        self.audio_length = Capturing.get_audio_length()

        self.audio_interface = pyaudio.PyAudio()

        self.mixer = self.get_audio_mixer_id()

        self.stream = self.audio_interface.open(
            format=self.audio_format,
            channels=2,
            rate=self.audio_sample_rate,
            frames_per_buffer=self.audio_sample_rate * self.audio_length,
            input=True,
            input_device_index=self.mixer,
        )

        self.killed = False

    def get_audio_mixer_id(self):
        if Capturing.get_audio_mixer_id() != -1:
            return Capturing.get_audio_mixer_id()

        for i in range(self.audio_interface.get_device_count()):
            dev = self.audio_interface.get_device_info_by_index(i)
            if 'Stereo Mix' in dev['name'] and dev['hostApi'] == 0:
                dev_index = dev['index']
                print('Stereo Mix Dev Index:', dev_index)
                return dev_index

    def get_audio(self):
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

        stream = self.stream.read(self.audio_sample_rate * self.audio_length)
        decoded = np.frombuffer(stream, dtype=audio_format)

        separate_channel = np.reshape(decoded, (self.audio_sample_rate * self.audio_length, 2))

        self.put_data((separate_channel[:, 0], separate_channel[:, 1]))

    def put_data(self, data):
        try:
            if self.audio.full():
                self.audio.get()

            self.audio.put(data)
        except FileNotFoundError:
            print("Pipeline dead.")
            self.killed = True

    def run(self):
        while True:
            if self.killed:
                break

            time.sleep(0.01)  # update the queue every 10ms
