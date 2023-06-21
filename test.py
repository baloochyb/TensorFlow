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

# t1 = tf.Variable(42)
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

# i = tf.cast(t1, dtype=tf.int32)
# j = tf.cast(tf.constant(4.9), dtype=tf.int32)
# print(i, '\n', j)

# ragged = tf.ragged.constant([[5, 2, 6, 1], [], [4, 10, 7], [8], [6,7]])
# print(ragged)
# print(ragged[0,:])
# print(ragged[1,:])
# print(ragged[2,:])
# print(ragged[3,:])
# print(ragged[4,:])

# print(tf.RaggedTensor.from_row_splits(values=[5, 2, 6, 1, 4, 10, 7, 8, 6, 7], row_splits=[0, 4, 4, 7, 8, 10]))

# x = [1,3,5,7,11]
# y = 5
# s = tf.math.squared_difference(x, y)
# print(s)

# numbers = tf.constant([[4., 5.], [7., 3.]])
#print(tf.reduce_mean(input_tensor=numbers))
# print(tf.reduce_mean(input_tensor=numbers, axis=0))
# print(tf.reduce_mean(input_tensor=numbers, axis=0, keepdims=True))
# print(tf.reduce_mean(input_tensor=numbers, axis=1))
# print(tf.reduce_mean(input_tensor=numbers, axis=1, keepdims=True))

# print(tf.random.normal(shape=(3, 2), mean=10.0, stddev=2.0, dtype=tf.float32, seed=None, name=None))
# print(tf.random.uniform(shape=(2, 4), minval=0, maxval=None, dtype=tf.float32, seed=None, name=None))

# tf.random.set_seed(11)
# ran1 = tf.random.uniform(shape = (2,2), maxval=10, dtype = tf.int32)
# ran2 = tf.random.uniform(shape = (2,2), maxval=10, dtype = tf.int32)
# print(ran1) #Call 1
# print(ran2)
# tf.random.set_seed(11) #same seed
# ran1 = tf.random.uniform(shape = (2,2), maxval=10, dtype = tf.int32)
# ran2 = tf.random.uniform(shape = (2,2), maxval=10, dtype = tf.int32)
# print(ran1) #Call 2
# print(ran2)

# dice1 = tf.Variable(tf.random.uniform(shape=[10, 1], minval=0, maxval=7, dtype=tf.int32))
# dice2 = tf.Variable(tf.random.uniform(shape=[10, 1], minval=1, maxval=7, dtype=tf.int32))
# # We may add dice1 and dice2 since they share the same shape and size.
# dice_sum = dice1 + dice2
# # We've got three separate 10x1 matrices. To produce a single
# # 10x3 matrix, we'll concatenate them along dimension 1.
# resulting_matrix = tf.concat(values=[dice1, dice2, dice_sum], axis=1)
# print(resulting_matrix)
