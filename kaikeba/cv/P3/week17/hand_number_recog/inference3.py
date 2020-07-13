#coding: utf-8
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import pdb

#im = Image.open('./mnist_test_image1.png')
#im = Image.open('./mnist_image2.png')
#im = Image.open('/jupyter_mnt/text.png')
im = Image.open('./hand_write/text.png')
im=im.resize((28,28),Image.ANTIALIAS)
im=np.array(im.convert('L'), 'f')
im=im.reshape([784])
print(im.size)
#pdb.set_trace()
#data = list(im.getdata())
data=im
print(data)
#result = [(255-list(x))*1.0/255.0 for x in data] 
result=data
print(result)

# 为输入图像和目标输出类别创建节点
x = tf.placeholder("float", shape=[None, 784]) # 训练所需数据  占位符

# *************** 构建多层卷积网络 *************** #
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1) # 取随机值，符合均值为0，标准差stddev为0.1
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x_image = tf.reshape(x, [-1,28,28,1]) # -1表示任意数量的样本数,大小为28x28，深度为1的张量

W_conv1 = weight_variable([5, 5, 1, 32]) # 卷积在每个5x5的patch中算出32个特征。
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) 
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 在输出层之前加入dropout以减少过拟合
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 全连接层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# 输出层
# tf.nn.softmax()将神经网络的输层变成一个概率分布
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

saver = tf.train.Saver() # 定义saver

# *************** 开始识别 *************** #
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./save/model.ckpt")#这里使用了之前保存的模型参数

    prediction = tf.argmax(y_conv,1)
    print("prediction",prediction)
    #predint = prediction.eval(feed_dict={x: [result],keep_prob: 1.0}, session=sess)
    predint = prediction.eval(feed_dict={x: [data],keep_prob: 1.0}, session=sess)
    print(predint)
    print("recognize result: %d" %predint[0])
