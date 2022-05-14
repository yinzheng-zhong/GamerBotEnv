from src.Model.data_pipeline import DataPipeline
import tensorflow as tf
from src.Helper.configs import Hardware as hw_config


physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if hw_config.get_gpu_id() == -1:
        tf.config.set_visible_devices(physical_devices[1:], 'GPU')
    else:
        tf.config.set_visible_devices(physical_devices[hw_config.get_gpu_id()], 'GPU')

    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.compat.v1.Session(config=config)

    try:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)
    except Exception as e:
        print(e)
        print('Not support mixed precision')

if __name__ == "__main__":
    dp = DataPipeline()
    while True:
        #dp.print_text()
        dp.make_batch()
        #print("\n")