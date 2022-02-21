import numpy as np
import tensorflow as tf
from Util.arc_cosine_kernel import Arc_Cosine_Kernel
from Util.random_relu_network import Single_Layer_Relu_Network
from Util.random_relu_network import Random_Relu_Network

import matplotlib.pyplot as plt

n_samples = 5000
n_dimenstion = 1000
x = np.random.normal(0, 1, (n_samples, n_dimenstion))
y = np.random.normal(0, 1, (n_samples, n_dimenstion))


ACOS_K = Arc_Cosine_Kernel(order=1)
ground_truth = ACOS_K.kernel_function(x, y)


width_list = [1, 10, 100, 1000, 10000, 100000]
results = []
for width in width_list:
    Random_Relu_Net = Single_Layer_Relu_Network(width)
    approx = tf.linalg.matmul(Random_Relu_Net.output(x), Random_Relu_Net.output(y), transpose_b=True)
    results.append(tf.keras.metrics.mean_squared_error(tf.reshape(ground_truth, (-1,)), tf.reshape(approx, (-1,))))

plt.yscale('log')
plt.xticks(np.arange(0, len(width_list), 1), width_list)
plt.scatter(np.arange(0, len(width_list), 1), results)
plt.show()