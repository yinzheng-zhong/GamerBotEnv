"""
The code encodes the keys from the keyboard to the on-hot encoding.
Also it decodes the keys from the on-hot encoding to the keys.
"""
from src.Helper.configs import Keys as KeysConfig
import numpy as np
import src.Helper.constance as const


class KeyMapping:
    def __init__(self):
        self.enabled_keys = KeysConfig.get_keys_enabled() + ['idle']
        self.dict_key_ind = {}

        index = 0
        for i, key in enumerate(self.enabled_keys):
            self.dict_key_ind[self.enabled_keys[i]] = index
            index += 1

            self.dict_key_ind[self.enabled_keys[i] + const.KEY_RELEASE_SUFFIX] = index
            index += 1

    def get_on_hot_mapping(self, key_str):
        if key_str is None:
            return self.get_default_mapping()

        vector = np.zeros(len(self.dict_key_ind), dtype=np.int8)

        try:
            index = self.dict_key_ind[key_str]
        except KeyError:
            print('Key not found, please and the key in the setting: ', key_str)
            index = self.dict_key_ind['idle']

        vector[index] = 1

        return vector

    def get_key_from_on_hot_mapping(self, on_hot_mapping):
        index = np.argmax(on_hot_mapping)
        return self.enabled_keys[index]

    def get_default_mapping(self):
        """
        This is just the idle mapping
        """
        vector = np.zeros(len(self.dict_key_ind), dtype=np.int8)
        vector[-1] = 1

        return vector

    def get_mapping_size(self):
        return len(self.get_default_mapping())
