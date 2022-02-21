import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Single_Layer_Relu_Network:
    def __init__(self, width):
        self.width = width
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.model = tf.keras.models.Sequential([tf.keras.layers.Dense(self.width, activation='relu', kernel_initializer=initializer)])

    def output(self, x):
        return tf.math.sqrt(2/self.width) * self.model(x)

    def kernel_function(self, x, y):
        return tf.linalg.matmul(self.output(x), self.output(y), transpose_b=True)

class Random_Relu_Network:
    def __init__(self, width_list):
        self.width_list = width_list
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        layers = []
        for width in self.width_list:
            layers.append(tf.keras.layers.Dense(width, activation='relu', kernel_initializer=initializer))
            layers.append(tf.keras.layers.Lambda(lambda x: x * tf.math.sqrt(2/width)))
        self.model = tf.keras.models.Sequential(layers)

    def output(self, x):
        return self.model(x)

    def kernel_function(self, x, y):
        return tf.linalg.matmul(self.output(x), self.output(y), transpose_b=True)