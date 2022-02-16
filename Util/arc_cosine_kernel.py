import numpy as np
import tensorflow as tf

class Arc_Cosine_Kernel:
    def __init__(self, order=1):
        self.order = order

    def angular_function(self, theta):
        if self.order == 0:
            return np.pi - theta
        elif self.order == 1:
            return tf.sin(theta) + (np.pi - theta) * tf.cos(theta)
        elif self.order == 2:
            return 3 * tf.sin(theta) * tf.cos(theta) + (np.pi - theta) * (
                   1 + 2 * tf.cos(theta) ** 2)

    def theta(self, x, y, x_norm, y_norm):
        numerator = tf.reduce_sum(x * y, axis=1)
        denumerator = tf.reduce_sum(tf.reshape(x_norm, (-1, 1)) * tf.reshape(y_norm, (-1, 1)), axis=1)
        theta = tf.acos(numerator / denumerator)
        return theta

    def kernel_function(self, x, y):
        x_norm = tf.norm(x, axis=1)
        y_norm = tf.norm(y, axis=1)
        theta = self.theta(x, y, x_norm, y_norm)
        J = self.angular_function(theta)
        return (1/np.pi) * tf.pow(x_norm, self.order) * tf.pow(y_norm, self.order) * J
