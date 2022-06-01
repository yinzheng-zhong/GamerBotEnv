import torch
import torch.nn as nn
from src.Model.NN.nn_pytorch import CNN

from src.Utils.key_mapping import KeyMapping


class NN(CNN):
    def __init__(self):
        super().__init__()

        km = KeyMapping()
        key_size = km.get_mapping_size()

        self.last_key = nn.Sequential(
            nn.Linear(key_size, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU()
        )

        self.combined = nn.Sequential(
            nn.Linear(sum(self.fc_input_size) + 16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, self.key_output_size),
        )

    def forward(self, screen, sound_l, sound_r, last_key):
        screen = self.screen(screen).view(-1, self.fc_input_size[0])
        sound_l = self.sound(sound_l).view(-1, self.fc_input_size[1])
        sound_r = self.sound(sound_r).view(-1, self.fc_input_size[2])
        last_key = self.last_key(last_key)
        combined = torch.cat((screen, sound_l, sound_r, last_key), 1)
        output = self.combined(combined)
        return output
