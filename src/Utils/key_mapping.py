"""
The code encodes the keys from the keyboard to the on-hot encoding.
Also it decodes the keys from the on-hot encoding to the keys.
"""
from src.Helper.config_reader import Keys as KeysConfig
import numpy as np


class KeyMapping:
    def __init__(self):
        self.enabled_keys = KeysConfig.get_keys_enabled()

        self.dict_key_ind = {key: index for index, key in enumerate(self.enabled_keys)}
        self.dict_key_ind['idle'] = len(self.enabled_keys)

    def get_on_hot_mapping(self, key_str):
        vector = np.zeros(len(self.dict_key_ind), dtype=np.int8)
        index = self.dict_key_ind[key_str]
        vector[index] = 1

        return vector
