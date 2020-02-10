"""
@description: 跟数据有关的函数库
"""


"""
import
"""
import numpy as np
import cv2
import torch
import os

def encode_gray_label(labels):
    """
    将标签图的灰度值转换成类别id
    注意：ignoreInEval为True的都当分类0处理
    @param labels: 标签灰度图
    """
    encoded_labels = np.zeros_like(labels)
    # 除了下面特意转换的，其余都属于类别0
    # 1
    encoded_labels[labels == 200] = 1
    encoded_labels[labels == 204] = 1
    encoded_labels[labels == 209] = 1
    # 2
    encoded_labels[labels == 201] = 2
    encoded_labels[labels == 203] = 2
    # 3
    encoded_labels[labels == 217] = 3
    # 4
    encoded_labels[labels == 210] = 4
    # 5
    encoded_labels[labels == 214] = 5
    # 6
    encoded_labels[labels == 220] = 6
    encoded_labels[labels == 221] = 6
    encoded_labels[labels == 222] = 6
    encoded_labels[labels == 224] = 6
    encoded_labels[labels == 225] = 6
    encoded_labels[labels == 226] = 6
    # 7
    encoded_labels[labels == 205] = 7
    encoded_labels[labels == 227] = 7
    encoded_labels[labels == 250] = 7
    return encoded_labels

def decode_gray_label(labels):
    """
    将类别id恢复成灰度值
    @params labels: shape=(h, w)
    """
    decoded_labels = np.zeros_like(labels, dtype=np.int8)
    # 1
    decoded_labels[labels == 1] = 204
    # 2
    decoded_labels[labels == 2] = 203
    # 3
    decoded_labels[labels == 3] = 217
    # 4
    decoded_labels[labels == 4] = 210
    # 5
    decoded_labels[labels == 5] = 214
    # 6
    decoded_labels[labels == 6] = 224
    # 7
    decoded_labels[labels == 7] = 227
    return decoded_labels

def decode_color_label(labels):
    """
    将类别id恢复成RGB值
    @params labels: shape=(h, w)
    """
    decoded_labels = np.zeros((3, labels.shape[0], labels.shape[1]), dtype=np.int8)
    # 1
    decoded_labels[0][labels == 1] = 220
    decoded_labels[1][labels == 1] = 20
    decoded_labels[2][labels == 1] = 60
    # 2
    decoded_labels[0][labels == 2] = 119
    decoded_labels[1][labels == 2] = 11
    decoded_labels[2][labels == 2] = 32
    # 3
    decoded_labels[0][labels == 3] = 220
    decoded_labels[1][labels == 3] = 220
    decoded_labels[2][labels == 3] = 0
    # 4
    decoded_labels[0][labels == 4] = 128
    decoded_labels[1][labels == 4] = 64
    decoded_labels[2][labels == 4] = 128
    # 5
    decoded_labels[0][labels == 5] = 190
    decoded_labels[1][labels == 5] = 153
    decoded_labels[2][labels == 5] = 153
    # 6
    decoded_labels[0][labels == 6] = 180
    decoded_labels[1][labels == 6] = 165
    decoded_labels[2][labels == 6] = 180
    # 7
    decoded_labels[0][labels == 7] = 178
    decoded_labels[1][labels == 7] = 132
    decoded_labels[2][labels == 7] = 190
    return decoded_labels


def crop_resize_data(image, labels, out_size, height_crop_offset):
    """
    @param out_size: (w, h)
    """
    roi_image = image[height_crop_offset:] # crop
    roi_image = cv2.resize(roi_image, out_size, interpolation=cv2.INTER_LINEAR)  # resize
    if labels is not None:
        roi_label = labels[height_crop_offset:]
        roi_label = cv2.resize(roi_label, out_size, interpolation=cv2.INTER_NEAREST)  # label必须用最近邻来，因为每个像素值是一个分类id
    else:
        roi_label = None
    return roi_image, roi_label

def train_data_generator(image_list, label_list, batch_size, out_size, height_crop_offset):
    """
    训练数据生成器
    :@param image_list: 图片文件的绝对地址
    :@param label_list: 标签文件的绝对地址
    :@param batch_size: 每批取多少张图片
    :@param image_size: 输出的图片尺寸
    :@param crop_offset: 在高度的方向上，将原始图片截掉多少
    """
    indices = np.arange(0, len(image_list))  # 索引
    out_images = []
    out_labels = []
    out_images_filename = []
    while True:  # 可以无限生成
        np.random.shuffle(indices)
        for i in indices:
            try:
                image = cv2.imread(image_list[i])
                labels = cv2.imread(label_list[i], cv2.IMREAD_GRAYSCALE)
            except:
                continue
            # crop & resize
            image, labels = crop_resize_data(image, labels, out_size, height_crop_offset)
            # encode
            labels = encode_gray_label(labels)

            out_images.append(image)
            out_labels.append(labels)
            out_images_filename.append(image_list[i])
            if len(out_images) == batch_size:
                out_images = np.array(out_images, dtype=np.float32)
                out_labels = np.array(out_labels, dtype=np.int64)
                # 转换成RGB
                out_images = out_images[:, :, :, ::-1]
                # 维度改成 (n, c, h, w)
                out_images = out_images.transpose(0, 3, 1, 2)
                # 归一化 -1 ~ 1
                out_images = out_images*2/255 - 1
                yield torch.from_numpy(out_images), torch.from_numpy(out_labels).long(), out_images_filename
                out_images = []
                out_labels = []
                out_images_filename = []

def test_data_generator(images_root, batch_size, out_size, height_crop_offset):
    """
    测试数据生成器
    :@param image_root: 测试图片文件的所在目录
    :@param batch_size: 每批最多取多少张图片
    :@param image_size: 输出的图片尺寸
    :@param crop_offset: 在高度的方向上，将原始图片截掉多少
    """
    # 遍历测试图片目录
    out_images = []
    out_images_filename = []
    for file in os.listdir(images_root):
        if not file.endswith('.jpg'):
            continue
        try:
            image = cv2.imread(os.path.join(images_root, file))
        except:
            continue
        # crop & resize
        image, _ = crop_resize_data(image, None, out_size, height_crop_offset)

        out_images.append(image)
        out_images_filename.append(file)

        if len(out_images) == batch_size:
            out_images = np.array(out_images, dtype=np.float32)
            # 转换成RGB
            out_images = out_images[:, :, :, ::-1]
            # 维度改成 (n, c, h, w)
            out_images = out_images.transpose(0, 3, 1, 2)
            # 归一化 -1 ~ 1
            out_images = out_images*2/255 - 1
            yield torch.from_numpy(out_images), out_images_filename
            out_images = []
            out_images_filename = []

def decodePredicts(predicts, out_size, height_pad_offset, mode='color'):
    """
    将推断的结果恢复成图片
    @param predicts: shape=(n, c, h, w)
    @param out_size: 恢复的尺寸 (w, h)
    @param height_pad_offset: 在高度维度上填充回多少
    @param mode: color | gray
    """
    # softmax
    predicts = np.argmax(predicts, axis=1)
    # reshape to (n, -1)
    n, h, w = predicts.shape
    predicts = predicts.reshape((n, -1))
    if mode == 'color':
        predicts = decode_color_label(predicts)
        predicts = predicts.reshape((3, n, h, w))
        predicts = predicts.transpose((1, 2, 3, 0)) # to (n, h, w, c)
        c = 3
    elif mode == 'gray':
        predicts = decode_gray_label(predicts)
        predicts.reshape((n, 1, h, w))
        predicts = predicts.transpose((0, 2, 3, 1)) # to (n, h, w, c)
        c = 1
    else:
        raise ValueError('mode supports: color / gary')

    # resize & pad (必须用最近邻)
    dsize = (out_size[0], out_size[1]-height_pad_offset)
    outs = []
    for i in range(n):
        out = np.zeros((out_size[1], out_size[0], c), dtype=np.uint8)
        out[height_pad_offset:] = cv2.resize(predicts[i], dsize, interpolation=cv2.INTER_NEAREST)  # label
        outs.append(out)
    return outs

