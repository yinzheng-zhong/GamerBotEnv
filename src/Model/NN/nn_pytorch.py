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
        super().__init__()
        self.key_output_size = key_config.get_key_mapping_size()

        self.input_screen_dim = (
            1,
            nn_config.get_screenshot_input_dim()[1],
            nn_config.get_screenshot_input_dim()[0]
        )  # (720, 1280)

        self.input_sound_dim = (
            1,
            constance.NN_SOUND_SPECT_INPUT_DIM[1],
            constance.NN_SOUND_SPECT_INPUT_DIM[0]
        )

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

        self.fc_input_size = self.calc_fc_input_size()

        self.combined = nn.Sequential(
            nn.Linear(sum(self.fc_input_size), 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, self.key_output_size),
        )

    def forward(self, screen, sound_l, sound_r):
        screen = self.screen(screen).view(-1, self.fc_input_size[0])
        sound_l = self.sound(sound_l).view(-1, self.fc_input_size[1])
        sound_r = self.sound(sound_r).view(-1, self.fc_input_size[2])
        combined = torch.cat((screen, sound_l, sound_r), 1)
        output = self.combined(combined)
        return output

    def calc_fc_input_size(self):
        screen = torch.zeros(self.input_screen_dim).unsqueeze(0)
        screen = self.screen(screen)
        screen = screen.shape[1]

        sound = torch.zeros(self.input_sound_dim).unsqueeze(0)
        sound = self.sound(sound)
        sound = sound.shape[1]

        return screen, sound, sound
