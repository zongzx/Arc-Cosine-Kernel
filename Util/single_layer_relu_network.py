import numpy as np
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Single_Layer_Relu_Network:
    def __init__(self, width):
        self.width = width
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.model = tf.keras.models.Sequential([tf.keras.layers.Dense(self.width, activation='relu', kernel_initializer=initializer)])

    def predict(self, x):
        return tf.math.sqrt(2/self.width) * self.model(x)
