import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import mixed_precision
import os

def setup_tensorflow():
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    # Enable GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Set TensorFlow GPU allocator environment variable
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def clear_session():
    K.clear_session()