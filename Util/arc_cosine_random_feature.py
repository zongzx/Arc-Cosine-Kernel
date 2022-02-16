import numpy as np
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

class Arc_Cosine_Random_Feature:
    def __init__(self, order, number_of_random_features, number_of_data_points):
        self.order = order
        self.number_of_random_features = number_of_random_features
        # type(number_of_random_features) is "list"
        # len(number_of_random_features) should be equal to network depth
        # number_of_random_features[i] is the number of random features at layer i
        self.depth = len(number_of_random_features)
        self.number_of_data_points = number_of_data_points
        self.random_vectors = []
        
    def generate_random_vectors(self):
        for layer in range(self.depth):
            if (layer == 0):
                one_layer_random_vectors = np.random.normal(size=(self.number_of_random_features, self.number_of_data_points))

            self.random_vectors.append()

