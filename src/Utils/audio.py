import wave
import librosa


def save_audio(audio, filename, sample_size=2, sample_rate=44100):
    """
    Save audio to file
    :param audio: bytes
    :param filename: str
    """

    wf = wave.open(filename, 'wb')
    wf.setnchannels(2)
    wf.setsampwidth(sample_size)
    wf.setframerate(sample_rate)
    wf.writeframes(audio)
    wf.close()


def mel_spectrogram_mono(audio, sample_rate=44100):
    """
    Compute mel-spectrogram
    """

    return librosa.feature.melspectrogram(y=audio, sr=sample_rate, power=4)
