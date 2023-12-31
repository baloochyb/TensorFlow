"""A simple TensorFlow application"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

# Create tensor
msg = tf.strings.join(["Hello ", "TensorFlow!"])

# Launch session
with tf.Session() as sess:
    print(sess.run(msg))
