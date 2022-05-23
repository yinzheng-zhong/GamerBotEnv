import tensorflow as tf
from tensorflow import keras
from src.Helper.configs import NN as nn_config
from src.Helper.configs import Keys as key_config
from src.Helper.configs import Hardware as hw_config
from src.Helper import constance


class NeuralNetwork:
    def __init__(self):
        self.time_steps = nn_config.get_time_steps()

        self.input_screen_dim = (
            nn_config.get_screenshot_input_dim()[1],
            nn_config.get_screenshot_input_dim()[0],
            1
        )  # (720, 1280)

        self.input_sound_dim = (
            constance.NN_SOUND_SPECT_INPUT_DIM[1],
            constance.NN_SOUND_SPECT_INPUT_DIM[0],
            1
        )  # (128, 128)
        self.key_output_size = key_config.get_key_mapping_size()

        model_type = nn_config.get_model_type()

        if model_type == constance.NN_MODEL_SINGLE:
            self.model = self.single_cnn()
        elif model_type == constance.NN_MODEL_LSTM:
            self.model = self.lstm()
        else:
            raise ValueError('Model not found')

    def single_cnn(self):
        input_screen = keras.Input(shape=self.input_screen_dim, name='x0')

        screen_conv_0 = keras.layers.Conv2D(16, (3, 3), activation='relu')(input_screen)
        screen_pool_0 = keras.layers.MaxPooling2D((2, 2))(screen_conv_0)
        screen_conv_1 = keras.layers.Conv2D(32, (3, 3), activation='relu')(screen_pool_0)
        screen_pool_1 = keras.layers.MaxPooling2D((2, 2))(screen_conv_1)
        screen_conv_2 = keras.layers.Conv2D(16, (3, 3), activation='relu')(screen_pool_1)
        screen_pool_2 = keras.layers.MaxPooling2D((2, 2))(screen_conv_2)
        screen_flat = keras.layers.Flatten()(screen_pool_2)

        input_sound_l = keras.Input(shape=self.input_sound_dim, name='x1')
        input_sound_r = keras.Input(shape=self.input_sound_dim, name='x2')

        input_feedback_action = keras.Input(shape=(self.key_output_size,), name='x3')
        input_feedback_cursor = keras.Input(shape=(2,), name='x4')

        sound_conv_l_0 = keras.layers.Conv2D(16, (3, 3), activation='relu')(input_sound_l)
        sound_pool_l_0 = keras.layers.MaxPooling2D((2, 2))(sound_conv_l_0)
        sound_conv_l_1 = keras.layers.Conv2D(16, (3, 3), activation='relu')(sound_pool_l_0)
        sound_pool_l_1 = keras.layers.MaxPooling2D((2, 2))(sound_conv_l_1)
        sound_flat_l = keras.layers.Flatten()(sound_pool_l_1)

        sound_conv_r_0 = keras.layers.Conv2D(16, (3, 3), activation='relu')(input_sound_r)
        sound_pool_r_0 = keras.layers.MaxPooling2D((2, 2))(sound_conv_r_0)
        sound_conv_r_1 = keras.layers.Conv2D(16, (3, 3), activation='relu')(sound_pool_r_0)
        sound_pool_r_1 = keras.layers.MaxPooling2D((2, 2))(sound_conv_r_1)
        sound_flat_r = keras.layers.Flatten()(sound_pool_r_1)

        feedback_action_dense = keras.layers.Dense(16, activation='relu')(input_feedback_action)
        feedback_cursor_dense = keras.layers.Dense(8, activation='relu')(input_feedback_cursor)

        concatenated = keras.layers.concatenate([
            screen_flat, sound_flat_l, sound_flat_r, feedback_action_dense, feedback_cursor_dense
        ])

        dense_0 = keras.layers.Dense(12, activation='relu')(concatenated)
        dense_0 = keras.layers.Dense(16, activation='relu')(dense_0)
        dense_1 = keras.layers.Dense(16, activation='relu')(dense_0)

        dense_out_1 = keras.layers.Dense(24, activation='relu')(dense_1)
        dense_out_2 = keras.layers.Dense(16, activation='relu')(dense_1)

        output_key = keras.layers.Dense(self.key_output_size, activation='linear', name='y0')(dense_out_1)
        output_cursor = keras.layers.Dense(2, activation='relu', name='y1')(dense_out_2)  # for cursor

        model = keras.Model(
            inputs=[input_screen, input_sound_l, input_sound_r, input_feedback_action, input_feedback_cursor],
            outputs=[output_key, output_cursor]
        )

        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['accuracy'])
        model.summary()

        return model

    def lstm(self):
        """
        lstm model. the audio part is the same as single_cnn
        """
        input_screen = keras.Input(shape=(self.time_steps, *self.input_screen_dim), name='x0')

        screen_conv_lstm_0 = keras.layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='relu',
                                                     return_sequences=True)(input_screen)
        screen_pool_lstm_0 = keras.layers.MaxPooling3D((1, 2, 2))(screen_conv_lstm_0)
        screen_conv_lstm_1 = keras.layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='relu')(
            screen_pool_lstm_0)
        screen_pool_lstm_1 = keras.layers.MaxPooling2D((2, 2))(screen_conv_lstm_1)

        screen_conv_0 = keras.layers.Conv2D(16, (3, 3), activation='relu')(screen_pool_lstm_1)
        screen_pool_0 = keras.layers.MaxPooling2D((2, 2))(screen_conv_0)
        screen_conv_1 = keras.layers.Conv2D(16, (3, 3), activation='relu')(screen_pool_0)
        screen_pool_1 = keras.layers.MaxPooling2D((2, 2))(screen_conv_1)
        screen_conv_2 = keras.layers.Conv2D(8, (3, 3), activation='relu')(screen_pool_1)
        screen_pool_2 = keras.layers.MaxPooling2D((2, 2))(screen_conv_2)
        screen_flat = keras.layers.Flatten()(screen_pool_2)

        input_sound_l = keras.Input(shape=self.input_sound_dim, name='x1')
        input_sound_r = keras.Input(shape=self.input_sound_dim, name='x2')

        input_feedback_action = keras.Input(shape=(self.time_steps, self.key_output_size), name='x3')
        input_feedback_cursor = keras.Input(shape=(self.time_steps, 2), name='x4')

        sound_conv_l_0 = keras.layers.Conv2D(16, (3, 3), activation='relu')(input_sound_l)
        sound_pool_l_0 = keras.layers.MaxPooling2D((2, 2))(sound_conv_l_0)
        sound_conv_l_1 = keras.layers.Conv2D(8, (3, 3), activation='relu')(sound_pool_l_0)
        sound_pool_l_1 = keras.layers.MaxPooling2D((2, 2))(sound_conv_l_1)
        sound_flat_l = keras.layers.Flatten()(sound_pool_l_1)

        sound_conv_r_0 = keras.layers.Conv2D(16, (3, 3), activation='relu')(input_sound_r)
        sound_pool_r_0 = keras.layers.MaxPooling2D((2, 2))(sound_conv_r_0)
        sound_conv_r_1 = keras.layers.Conv2D(8, (3, 3), activation='relu')(sound_pool_r_0)
        sound_pool_r_1 = keras.layers.MaxPooling2D((2, 2))(sound_conv_r_1)
        sound_flat_r = keras.layers.Flatten()(sound_pool_r_1)

        feedback_action_dense = keras.layers.LSTM(16, activation='relu')(input_feedback_action)
        feedback_cursor_dense = keras.layers.LSTM(8, activation='relu')(input_feedback_cursor)

        concatenated = keras.layers.concatenate([
            screen_flat, sound_flat_l, sound_flat_r, feedback_action_dense, feedback_cursor_dense
        ])

        dense_0 = keras.layers.Dense(16, activation='relu')(concatenated)
        dense_1 = keras.layers.Dense(32, activation='relu')(dense_0)

        dense_out_1 = keras.layers.Dense(32, activation='relu')(dense_1)
        dense_out_2 = keras.layers.Dense(16, activation='relu')(dense_1)

        output_key = keras.layers.Dense(self.key_output_size, activation='linear', name='y0')(dense_out_1)
        output_cursor = keras.layers.Dense(2, activation='relu', name='y1')(dense_out_2)  # for cursor

        model = keras.Model(
            inputs=[input_screen, input_sound_l, input_sound_r, input_feedback_action, input_feedback_cursor],
            outputs=[output_key, output_cursor]
        )

        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['accuracy'])
        model.summary()

        return model

