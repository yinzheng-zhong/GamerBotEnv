import tensorflow as tf
from tensorflow import keras


class NeuralNetwork:
    def __init__(self, input_size, output_size, model='single_cnn'):
        self.input_size = input_size
        self.output_size = output_size

        if model == 'single_cnn':
            self.model = self.single_cnn()
        else:
            raise ValueError('Model not found')

    def single_cnn(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_size))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(10, activation='softmax'))
        return model
