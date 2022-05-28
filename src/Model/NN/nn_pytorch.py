"""
PyTorch implementation of the neural networks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Helper.configs import NN as nn_config
from src.Helper import constance
from src.Helper.configs import Keys as key_config


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.key_output_size = key_config.get_key_mapping_size()

        self.screen = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (3, 3), stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 16, (3, 3), stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),
        )

        self.sound = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 16, (3, 3), stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),
        )

        self.combined = nn.Sequential(
            nn.Linear(38336, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, self.key_output_size),
        )

    def forward(self, screen, sound_l, sound_r):
        screen = self.screen(screen).view(-1, 22848)
        sound_l = self.sound(sound_l).view(-1, 7744)
        sound_r = self.sound(sound_r).view(-1, 7744)
        combined = torch.cat((screen, sound_l, sound_r), 1)
        output = self.combined(combined)
        return output


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.key_output_size = key_config.get_key_mapping_size()

        self.screen = nn.Sequential()