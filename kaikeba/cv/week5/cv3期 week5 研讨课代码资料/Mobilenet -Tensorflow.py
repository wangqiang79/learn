import keras
import numpy as np
import math
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

#一般使用的都是layer层
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D,ReLU

from keras import backend as K
from keras import optimizers, regularizers
from keras.callbacks import LearningRateScheduler, TensorBoard

#Load Data
def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]

    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test


###使用cifar10来load data
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)

#调用上面的函数
x_train,x_test=color_preprocessing(x_train,x_test)

def block(inputs, out_chans, k, s):
    x = Conv2D(out_chans,k,padding='same',strides=s)(inputs)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    return x

def inverted_residual_block(inputs, out_chans, k, t, s, r=False):
    #tensoflow:nhwc pytorch:nchw
    #gpu运算 nvidia cudnn nchw tf是cpu进行过优化
    #输入的最后一维就是channel，然后放大t倍，这就是输出的channel
    tchannel = K.int_shape(inputs)[-1] * t

    #使用了1*1的kernel进行升维，升到t channel
    x = block(inputs,tchannel,1,1)

    #查一下depth_multiplier用法，一般是设置为1
    x = DepthwiseConv2D(k,strides=s,depth_multiplier=1,padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)

    x = Conv2D(out_chans,(1,1),strides=(1,1),padding='same')(x)
    x = BatchNormalization()(x)

    if r:
        x = Add()([x,inputs])
    return x

def make_layer(inputs, out_chans, k, t, s, n):
    x = inverted_residual_block(inputs,out_chans,k,t,s)
    for i in range(1,n):
        x = inverted_residual_block(x,out_chans,k,t,1,True)
    return x


def MobileNetv2(input_shape, k):

    inputs = Input(shape=input_shape)
    x = block(inputs,32,(3,3),s=(2,2))

    x = inverted_residual_block(x,16,(3,3),t=1,s=1)

    x = make_layer(x,24,(3,3),t=6,s=2,n=2)
    x = make_layer(x, 32, (3, 3), t=6, s=2, n=2)
    x = make_layer(x, 64, (3, 3), t=6, s=2, n=2)
    x = make_layer(x, 96, (3, 3), t=6, s=2, n=2)
    x = make_layer(x, 160, (3, 3), t=6, s=2, n=2)

    x = inverted_residual_block(x,320,(3,3),t=6,s=1)

    x = block(x,1280,(1,1),s=(1,1))
    x = GlobalAveragePooling2D()(x)

    x = Reshape(1,1,1280)(x)
    x = Dropout(0.3,name='Dropout')(x)
    x = Conv2D(k,(1,1),padding='same')(x)

    x = Activation('softmax',name='softmax')(x)
    output = Reshape((k,))(x)

    model = Model(inputs,output)
    return model

#TensorFlow中数据生成，优化器的设置，包括训练过程
#学习率的设置
def scheduler(epoch):
    if epoch < 150:
        return 0.1
    if epoch < 225:
        return 0.01
    return 0.001

# set optimizer
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)

#model编译
#不同的任务有不同的loss方法
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#TensorBoard是一个可视化工具
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)

#学习率使用调度
change_lr = LearningRateScheduler(scheduler)

cbks = [change_lr,tb_cb]

datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125,
                             height_shift_range=0.125,
                             fill_mode='constant',cval=0.)
datagen.fit(x_train)

# start training
#通过flow将data进行输入
model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                    steps_per_epoch=iterations,
                    epochs=epochs,
                    callbacks=cbks,
                    validation_data=(x_test, y_test))

model.save('mobilenetv2.h5')
model.summary()


