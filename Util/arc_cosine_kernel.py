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
        # Only work when self.order == 1
        numerator = tf.linalg.matmul(x, y, transpose_b=True)
        denumerator = x_norm * y_norm
        jitter = 1e-15
        theta = tf.acos((1 - 2 * jitter) * numerator / denumerator)
        return theta

    def kernel_function(self, x, y):
        n_samples_x = x.shape[0]
        n_samples_y = y.shape[0] 
        x_norm = tf.reshape(tf.norm(x, axis=1), (n_samples_x, 1))
        x_norm = tf.repeat(x_norm, n_samples_y, axis=1)
        y_norm = tf.reshape(tf.norm(y, axis=1), (1, n_samples_y))
        y_norm = tf.repeat(y_norm, n_samples_x, axis=0)
        theta = self.theta(x, y, x_norm, y_norm)
        J = self.angular_function(theta)
        return (1/np.pi) * tf.pow(x_norm, self.order) * tf.pow(y_norm, self.order) * J
