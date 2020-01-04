#导入所有的依赖包
import  tensorflow as tf
import numpy as np
from cnnModel import cnnModel
import os
import pickle
import getConfig
import sys
#初始化一个字典，用于存储从配置文件中读取的参数配置
gConfig = {}
#使用get_config方法获取配置文件中的参数
gConfig=getConfig.get_config(config_file="config.ini")
#定义read_data函数，读取数据
def read_data(dataset_path, im_dim, num_channels,num_files,images_per_file):
         # 获取训练集中训练文件的名称
        files_names = os.listdir(dataset_path)
        print(files_names)
        #创建空的多维数组用于存放图片的二进制数据
        dataset_array = np.zeros(shape=(num_files * images_per_file, im_dim, im_dim, num_channels))
        # 创建空的数组用于存放图片的标注信息
        dataset_labels = np.zeros(shape=(num_files * images_per_file), dtype=np.uint8)
        index = 0
        #从训练集中读取二进制数据并将其维度转换成32*32*3
        for file_name in files_names:
            if file_name[0:len(file_name)-1] == "data_batch_":
                print("正在处理数据 : ", file_name)
                data_dict = unpickle_patch(dataset_path + file_name)
                images_data = data_dict[b"data"]
                print(images_data.shape)
                # 格式转换为32x32x3 shape.
                images_data_reshaped = np.reshape(images_data, newshape=(len(images_data), im_dim, im_dim, num_channels))
                # 将维度转换后的图片数据存入指定数组内
                dataset_array[index * images_per_file:(index + 1) * images_per_file, :, :, :] = images_data_reshaped
                #  将维度转换后的标注数据存入指定数组内
                dataset_labels[index * images_per_file:(index + 1) * images_per_file] = data_dict[b"labels"]
                index = index + 1
        return dataset_array, dataset_labels  # 返回数据

def unpickle_patch(file):
    #打开文件，读取二进制文件，返回读取到的数据
    patch_bin_file = open(file, 'rb')
    patch_dict = pickle.load(patch_bin_file, encoding='bytes')
    return patch_dict

def create_model():
    #判断是否有预训练模型
    if 'pretrained_model'in gConfig:
        model=tf.keras.models.load_model(gConfig['pretrained_model'])
        return model
    ckpt=tf.io.gfile.listdir(gConfig['working_directory'])

    #判断是否已经有model文件存在，如果model文件存在则加载原来的model并在原来的moldel继续训练，如果不存在则新建model相关文件
    if  ckpt:
        #如果存在模型文件，则加载存放model文件夹中最新的文件
        model_file=os.path.join(gConfig['working_directory'], ckpt[-1])
        print("Reading model parameters from %s" % model_file)
        #使用tf.keras.models.load_model来加载模型

        model=tf.keras.models.load_model(model_file)
        return model
    else:
        model=cnnModel(gConfig['keeps'])
        model=model.createModel()
        return model

#使用read_data函数读取训练数据
dataset_array, dataset_labels = read_data(dataset_path=gConfig['dataset_path'], im_dim=gConfig['im_dim'],
   num_channels=gConfig['num_channels'],num_files=5,images_per_file=gConfig['images_per_file'])
#使用read_data函数读取训练数据
test_array, test_labels = read_data(dataset_path=gConfig['test_path'], im_dim=gConfig['im_dim'],
   num_channels=gConfig['num_channels'],num_files=1,images_per_file=gConfig['images_per_file'])
#对读取到的输入数据进行归一化处理
dataset_array=dataset_array.astype('float32')/255
test_array=test_array.astype('float32')/255
#将读取到的标注数据进行Onehot编码
dataset_labels=tf.keras.utils.to_categorical(dataset_labels,10)
test_labels=tf.keras.utils.to_categorical(test_labels,10)
#定义训练函数
def train():
    #初始化Model
     model=create_model()
     print(model.summary())
    #使用fit方法进行训练，verbose是控制输出的信息，validation_data是配置测试数据集
     model.fit(dataset_array,dataset_labels,verbose=1,epochs=gConfig['epochs'],validation_data=(test_array,test_labels))

     # 达到一个训练模型保存点后，将模型保存下来，并打印出这个保存点的平均准确率
     filename='cnn_model.h5'
     checkpoint_path = os.path.join(gConfig['working_directory'], filename)
     model.save(checkpoint_path)
     sys.stdout.flush()
#定义预测函数
def predict(data):
    #获取batches数据，也就是不同种类对应的名称
    file = gConfig['dataset_path'] + "batches.meta"
    patch_bin_file = open(file, 'rb')
    #最终生成一个字典，以便将预测结果转换为具体物体的名称
    label_names_dict = pickle.load(patch_bin_file)["label_names"]
    #初始化model
    model=create_model()
    #使用predict方法对输入数据进行预测
    predicton=model.predict(data)
    #使用argmax将预测结果转换为对应的数字索引
    index=tf.math.argmax(predicton[0]).numpy()
    #返回预测结果的名称
    return label_names_dict[index]

if __name__=='__main__':
    #如果是训练模式则直接调用train函数进行训练
    if gConfig['mode']=='train':
        train()
    #如果是服务模式，在使用app.py
    elif gConfig['mode']=='serve':
        print('请使用:python3 app.py')

    
