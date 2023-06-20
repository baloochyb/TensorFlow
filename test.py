"""Docstring"""
import tensorflow as tf

# print("TensorFlow version: {}".format(tf.__version__))
# print("Eager execution is: {}".format(tf.executing_eagerly()))
# print("Keras version: {}".format(tf.keras.__version__))

# var = tf.Variable([3, 3])

# print(tf.config.list_physical_devices('GPU'))
# if tf.test.is_gpu_available():
#     print('Running on GPU')
#     print('GPU #0?')
#     print(var.device.endswith('GPU:0'))
# else:
#     print('Running on CPU')

t1 = tf.Variable(42)
# t2 = tf.Variable([ [ [0., 1., 2.], [3., 4., 5.] ], [ [6., 7., 8.], [9., 10., 11.] ] ])
# print(t1, t2)

# f64 = tf.Variable(89, dtype=tf.float64)
# print(f64)
# f64.assign(98.)
# print(f64)

# m_o_l = tf.constant(42)
# print(m_o_l)
# print(m_o_l.numpy())
# unit = tf.constant(1, dtype = tf.int64)
# print(unit)

# t2 = tf.Variable([ [ [0., 1., 2.], [3., 4., 5.] ], [ [6., 7., 8.], [9., 10., 11.] ] ])
# print(t2.shape)

# r1 = tf.reshape(t2, [2, 6])
# r2 = tf.reshape(t2, [1, 12])
# print(r1)
# print(r2)

# print(tf.rank(t2))
# t3 = t2[1, 0, 2] # slice 1, row 0, column 2
# print(t3)

# print(t2.numpy())
# print(t2[1, 0, 2].numpy())
# s = tf.size(input=t2).numpy()
# print(s)
# print(t2.dtype)
# print(t2*t2)
# print(t2*4)
# u = tf.constant([[3, 4, 3]])
# v = tf.constant([[1,2,1]])
# print(tf.matmul(u, tf.transpose(a=v)))

i = tf.cast(t1, dtype=tf.int32)
j = tf.cast(tf.constant(4.9), dtype=tf.int32)
print(i, '\n', j)
