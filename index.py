import tensorflow as tf
import numpy as np
import json
import os
import re


model_path = "./model_/600"
data_path = "/var/dataset/2020-6-27.dat"


sec_block = 600
cell = 40
batch_size = 2

steps = 60*60*24*3
show_loss_steps = 60*60 

# 添加神经层封装函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs



data = []
for line in open(data_path):
    data.append(re.findall(r"\d+\.?\d*",line))


x = []
y = []

for item in data:
    x.append(item[:22])
    y.append(item[-4:])


x_data = []
y_data = []

for index in range(len(x) - sec_block - 2):
    tmp_sec_block = sec_block
    tmp_array = []
    while tmp_sec_block > 0:
        tmp_array += x[index + tmp_sec_block - 1]
        tmp_sec_block = tmp_sec_block - 1
    x_data.append(tmp_array)



for index in range(len(y) - sec_block - 2):
    y_data.append(y[index + sec_block - 1])




 # 构建所需的数据


    # 定义占位符输入变量
xs = tf.placeholder(tf.float32, [None, 22*sec_block])
ys = tf.placeholder(tf.float32, [None, 4])

    ############### 搭建网络 ###############

    # 输入层1个，隐藏层10个，激励函数relu
l1 = add_layer(xs, 22*sec_block, cell, activation_function=tf.nn.relu)

    # 输输入层10个，输出层1个，无激励函数
prediction = add_layer(l1, cell, 4, activation_function=None)

    # 计算预测值prediction和真实值的误差，对二者差的平方求和再取平均。
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
            reduction_indices=[1]))

    # 接下来，是很关键的一步，如何让机器学习提升它的准确率。
    # tf.train.GradientDescentOptimizer()中的值通常都小于1，
    # 这里取的是0.1，代表以0.1的效率来最小化误差loss。
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始化
init = tf.global_variables_initializer()
sess = tf.Session()
saver = tf.train.Saver()
sess.run(init)

#saver.restore(sess, model_path)

    # 比如这里，我们让机器学习1000次。机器学习的内容是train_step,
    # 用 Session 来 run 每一次 training 的数据，逐步提升神经网络的预测准确性。
    # (注意：当运算要用到placeholder时，就需要feed_dict这个字典来指定输入。)
for i in range(steps):
    # training
    start = (i * batch_size) % len(x_data)
    end = start + batch_size
    sess.run(train_step, feed_dict={xs: x_data[start:end], ys: y_data[start:end]})
    if i % show_loss_steps == 0:
        # 每50步我们输出一下机器学习的误差。
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

saver.save(sess, model_path)
print('---------')
#print(sess.run(prediction, feed_dict={xs: x_data[0]}))
