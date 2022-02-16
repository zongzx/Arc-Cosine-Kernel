import numpy as np
import tensorflow as tf
from Util.arc_cosine_kernel import Arc_Cosine_Kernel
from Util.single_layer_relu_network import Single_Layer_Relu_Network

import matplotlib.pyplot as plt

n_samples = 5000
n_dimenstion = 10
x = np.random.uniform(-10, 10, (n_samples, n_dimenstion))
y = np.random.uniform(-10, 10, (n_samples, n_dimenstion))

ACOS_K = Arc_Cosine_Kernel(order=1)
ground_truth = ACOS_K.kernel_function(x, y)

#Random_Relu_Net = Single_Layer_Relu_Network(width=10000)

width_list = [1, 10, 100, 1000, 10000, 100000]
results = []
for width in width_list:
    Random_Relu_Net = Single_Layer_Relu_Network(width=width)
    approx = tf.reduce_sum( Random_Relu_Net.predict(x) * Random_Relu_Net.predict(y), axis=1)
    results.append(tf.keras.metrics.mean_squared_error(ground_truth, approx))

plt.plot(results)
plt.show()