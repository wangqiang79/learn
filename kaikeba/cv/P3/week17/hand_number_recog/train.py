#coding: utf-8
# Python3
# 使用LeNet5的七层卷积神经网络用于MNIST手写数字识别
import pdb
import tensorflow as tf
#import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#from tensorflow.examples.tutorials.mnist import input_data
from mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 为输入图像和目标输出类别创建节点
x = tf.placeholder(tf.float32, shape=[None, 784]) # 训练所需数据  占位符
y_ = tf.placeholder(tf.float32, shape=[None, 10]) # 训练所需标签数据  占位符

# *************** 构建多层卷积网络 *************** #

# 权重、偏置、卷积及池化操作初始化,以避免在建立模型的时候反复做初始化操作
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1) # 取随机值，符合均值为0，标准差stddev为0.1
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# x 的第一个参数为图片的数量，第二、三个参数分别为图片高度和宽度，第四个参数为图片通道数。
# W 的前两个参数为卷积核尺寸，第三个参数为图像通道数，第四个参数为卷积核数量
# strides为卷积步长，其第一、四个参数必须为1，因为卷积层的步长只对矩阵的长和宽有效
# padding表示卷积的形式，即是否考虑边界。"SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# x 参数的格式同tf.nn.conv2d中的x，ksize为池化层过滤器的尺度，strides为过滤器步长
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#把x更改为4维张量，第1维代表样本数量，第2维和第3维代表图像长宽， 第4维代表图像通道数
x_image = tf.reshape(x, [-1,28,28,1]) # -1表示任意数量的样本数,大小为28x28，深度为1的张量

# 第一层：卷积
W_conv1 = weight_variable([5, 5, 1, 32]) # 卷积在每个5x5的patch中算出32个特征。
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# 第二层：池化
h_pool1 = max_pool_2x2(h_conv1)

# 第三层：卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# 第四层：池化
h_pool2 = max_pool_2x2(h_conv2)

# 第五层：全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 在输出层之前加入dropout以减少过拟合
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第六层：全连接层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# 第七层：输出层
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# *************** 训练和评估模型 *************** #

# 为训练过程指定最小化误差用的损失函数，即目标类别和预测类别之间的交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

# 使用反向传播，利用优化器使损失函数最小化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 检测我们的预测是否真实标签匹配(索引位置一样表示匹配)
# tf.argmax(y_conv,dimension), 返回最大数值的下标 通常和tf.equal()一起使用，计算模型准确度
# dimension=0 按列找  dimension=1 按行找
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

# 统计测试准确率， 将correct_prediction的布尔值转换为浮点数来代表对、错，并取平均值。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver() # 定义saver

# *************** 开始训练模型 *************** #
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(3000):
      batch = mnist.train.next_batch(50)
      #pdb.set_trace()
      if i%100 == 0:
        # 评估模型准确度，此阶段不使用Dropout
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))

      # 训练模型，此阶段使用50%的Dropout
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    saver.save(sess, './save/model.ckpt') #模型储存位置

    print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images [0:2000], y_: mnist.test.labels [0:2000], keep_prob: 1.0}))
