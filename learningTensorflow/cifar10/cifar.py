import cifar10_input
import tensorflow as tf
import numpy as np
import pylab

#取数据
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'
print("begin")

images_train,label_train = cifar10_input.inputs(eval_data = False , data_dir = data_dir,batch_size = batch_size)
images_test,label_test = cifar10_input.inputs(eval_data = True , data_dir = data_dir,batch_size = batch_size)
print("begin data")

#定义卷积网络
#权重初始化定义
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)  #权重初始化为标准差为0.1的随机数
    return tf.Variable(initial)
#误差初始化定义
def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)
#同卷积层
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding = 'SAME')
#池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
def avg_pool_6x6(x):
    return tf.nn.avg_pool(x,ksize=[1,6,6,1],strides = [1,6,6,1],padding = 'SAME')

#占位符定义
x = tf.placeholder(tf.float32,[None,24,24,3])  #输入图像尺寸24x24x3
y = tf.placeholder(tf.float32,[None,10])       #类别个数10
#卷积核参数定义
W_conv1 = weight_variable([5,5,3,64])
b_conv1 = bias_variable([64])
#reshape函数：shape：-1代表自动计算此维度->tensor变换为规定大小的数据
x_image = tf.reshape(x,[-1,24,24,3])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,64,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5,5,64,10])
b_conv3 = bias_variable([10])
h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3)+b_conv3)

nt_hpool3 = avg_pool_6x6(h_conv3)
nt_hpool13_flat = tf.reshape(nt_hpool3,[-1,10])
y_conv = tf.nn.softmax(nt_hpool13_flat)

#定义损失函数
cross_entropy = -tf.reduce_sum(y*tf.log(y_conv))

#定义优化器
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
tf.summary.scalar('loss',cross_entropy)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
tf.summary.scalar('accuracy',accuracy)
saver = tf.train.Saver()
model_path = "log/cifar10model.ckpt"
#开始训练
sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)
merged_summary_op2 = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('log/cafir10_with_summarys',sess.graph)
for i in range(15000):
    image_batch, label_batch = sess.run([images_train, label_train])
    label_b = np.eye(10, dtype=float)[label_batch]  # one hot

    train_step.run(feed_dict={x: image_batch, y: label_b}, session=sess)

    if i % 200 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: image_batch, y: label_b}, session=sess)
        print("step %d, training accuracy %g" % (i, train_accuracy))
    save_path = saver.save(sess,model_path)
    summary_str = sess.run(merged_summary_op2,feed_dict={x: image_batch, y: label_b})
    summary_writer.add_summary(summary_str,i)

image_batch, label_batch = sess.run([images_test, label_test])
label_b = np.eye(10, dtype=float)[label_batch]  # one hot
print("finished！ test accuracy %g" % accuracy.eval(feed_dict={
    x: image_batch, y: label_b}, session=sess))