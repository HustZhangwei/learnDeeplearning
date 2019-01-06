import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pylab
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#程序用于使用全连接层将图片分类
#定义网络参数
learning_rate = 0.01 #学习率
training_epochs = 25 #迭代次数
batch_size = 100     #训练批次
display_step = 1



#定义网络模型参数
n_hidden_1 = 256      #第一个隐藏层的节点个数
n_hidden_2 = 256      #第二个隐藏层的节点个数
n_input= 784          #数据集的维数为784维
n_classes = 10        #数据集的类别个数

#定义网络结构
#定义占位符
x = tf.placeholder("float",[None,n_input])
y = tf.placeholder("float",[None,n_classes])

#创建网络模型
def multilayer_perceptron(x,weights,biases):
    #第一层隐藏层
    layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    #第二层隐藏层
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    #输出层
    out_layer = tf.matmul(layer_2,weights['out'])+biases['out']
    return out_layer
#各层参数定义及初始化
weights = {
    'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}
biases = {
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

#输出值定义
pred = multilayer_perceptron(x,weights,biases)

#定义损失函数和优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred,labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
saver = tf.train.Saver()
model_path = "log/FCmodel.ckpt"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)

            _, c = sess.run([optimizer,cost],feed_dict = {x:batch_xs,y:batch_ys})
            avg_cost += c/total_batch

        if (epoch+1)%display_step == 0:
            print("Epoch:",'%04d'%(epoch+1),"cost=","{:.9f}".format(avg_cost))
    print("finished")

    #保存模型
    save_path = saver.save(sess,model_path)
    print("Model saved in file:%s"% save_path)

#读取模型
print("Staring 2nd session...")
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    saver.restore(sess2,model_path)

    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))

    #计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
    output = tf.argmax(pred,1)
    batch_xs,batch_ys = mnist.train.next_batch(2)
    outputval,predv = sess2.run([output,pred],feed_dict={x:batch_xs})
    print(outputval,predv,batch_ys)

    im = batch_xs[0]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()