import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pylab

#定义模拟数据
img = tf.Variable(tf.constant(1.0,shape = [1,4,4,1]))
filter = tf.Variable(tf.constant([1.0,0,-1,-2],shape = [2,2,1,1]))
#print("img = ",img,"filter = ",filter)
conv = tf.nn.conv2d(img,filter,strides = [1,2,2,1],padding = 'VALID')
cons = tf.nn.conv2d(img,filter,strides = [1,2,2,1],padding = 'SAME')
print(conv.shape)
print(cons.shape)

#反卷积操作
contv = tf.nn.conv2d_transpose(conv,filter,[1,4,4,1],strides = [1,2,2,1],padding = 'VALID')
conts = tf.nn.conv2d_transpose(cons,filter,[1,4,4,1],strides = [1,2,2,1],padding = 'SAME')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run([conv,filter]))
    print(sess.run([cons]))
    print(sess.run([contv]))
    print(sess.run([conts]))
