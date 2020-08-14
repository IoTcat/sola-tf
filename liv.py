#-*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import re
import random

#mnist = input_data.read_data_sets("MNIST_data/")

model_path = "./model_liv/180"
data_path = "/var/dataset/2020-6-27.dat"
test_path = "/var/dataset/2020-6-29.dat"



# 训练参数
n_epoches = 50
batch_size = 150
Learning_rate = 0.001
# 网络参数，把28x28的图片数据拆成28行的时序数据喂进RNN
n_inputs = 22
n_steps = 180 # 追溯秒数
n_hiddens = 150
n_outputs = 2  # 10分类

start_from = 18
end_from = 22



# 输入tensors
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

# 构建RNN结构
basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hiddens, state_is_tuple=True)
#basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_hiddens)
#basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_hiddens)  # 另一种创建基本单元的方式
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

# 前向传播，定义损失函数、优化器
logits = tf.layers.dense(states[-1], n_outputs)  # 与states tensor连接的全连接层，LSTM时为states[-1]，即h张量
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(cross_entropy)
#loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction)

optimizer = tf.train.AdamOptimizer(learning_rate=Learning_rate)
prediction = tf.nn.in_top_k(logits, y, 1)
train_op = optimizer.minimize(loss)

accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))  # cast函数将tensor转换为指定类型

# 从MNIST中读取数据
#X_test = mnist.test.images.reshape([-1, n_steps, n_inputs])
#y_test = mnist.test.labels




def getData(path):

    data = []
    for line in open(path):
        tmp = re.findall(r"\d+\.?\d*",line)
        if int(tmp[20]) >= start_from:
            if int(tmp[20]) <= end_from:
                data.append(tmp)


    x = []
    y = []

    for item in data:
        x.append(item[:22])
        y.append(item[-4:])


    x_data = []
    y_data = []


    for index in range(len(x) - n_steps - 2):
        tmp_sec_block = n_steps
        tmp_array = []
        while tmp_sec_block > 0:
            tmp_array.append(x[index + tmp_sec_block - 1])
            tmp_sec_block = tmp_sec_block - 1
        x_data.append(tmp_array)



    for index in range(len(y) - n_steps - 2):
        tmp = y[index + n_steps - 1]
        tmp = list(map(int, tmp))
        y_data.append(tmp[3])

    return x_data, y_data






# 训练阶段
init = tf.global_variables_initializer()
saver = tf.train.Saver()
loss_list = []
accuracy_list = []


with tf.Session() as sess:
    sess.run(init)
    
    saver.restore(sess, model_path)
    x_b, y_b = getData(data_path)
    x_t, y_t = getData(test_path)
    #print(len(x_b),'      ', len(y_b),'         ', len(x_t),'            ', len(y_t))
    n_batches = len(x_b) // batch_size  # 整除返回整数部分
    n_test = len(x_t) // batch_size
    # print("Batch_number: {}".format(n_batches))
    for epoch in range(n_epoches):
        a_b = 0
        a_t = 0
        for iteration in range(min(n_batches, n_test)):
            #X_batch, y_batch = mnist.train.next_batch(batch_size)
            #X_batch = X_batch.reshape([-1, n_steps, n_inputs])
            X_batch = x_b[iteration * batch_size : (iteration + 1) * batch_size] 
            y_batch = y_b[iteration * batch_size : (iteration + 1) * batch_size] 
            X_test = x_t[iteration * batch_size : (iteration + 1) * batch_size] 
            y_test = y_t[iteration * batch_size : (iteration + 1) * batch_size] 
            sess.run(train_op, feed_dict={X: X_batch, y: y_batch})
            loss_train = loss.eval(feed_dict={X: X_batch, y: y_batch})
            loss_list.append(loss_train)
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
            accuracy_list.append(acc_test)
            a_b = a_b + acc_train
            a_t = a_t + acc_test
            #print(epoch, '-', X_batch[0][0][20], '  ', "Train accuracy: {:.3f}".format(acc_train), "Test accuracy: {:.3f}".format(acc_test))

        print(epoch, '-', X_batch[0][0][20], '  ', "Train accuracy: {:.3f}".format(a_b / min(n_batches, n_test)), "Test accuracy: {:.3f}".format(a_t / min(n_batches, n_test)))
        
        saver.save(sess, model_path)


