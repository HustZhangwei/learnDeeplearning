import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epoch = 5
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length

def generateData():
    x = np.array(np.random.choice(2,total_series_length,p = [0.5,0.5]))
    y = np.roll(x,echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size,-1))
    y = y.reshape((batch_size,-1))

    return x,y
batchX_placeholder = tf.placeholder(tf.float32,[batch_size,truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32,[batch_size,truncated_backprop_length])
init_state = tf.placeholder(tf.float32,[batch_size,state_size])

input_series = tf.unstack(batchX_placeholder,axis = 1)
labels_series = tf.unstack(batchY_placeholder,axis = 1)