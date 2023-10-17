import tensorflow as tf

gpu_devices = tf.config.list_logical_devices('GPU')
print(gpu_devices)