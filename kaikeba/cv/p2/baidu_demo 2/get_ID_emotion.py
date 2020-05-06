'''
  get face_id and face emotion
'''

from __future__ import print_function
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import cv2

from light_cnn.light_cnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
from models import mobilenetv3

parser = argparse.ArgumentParser(description='PyTorch ImageNet Feature Extracting')
parser.add_argument('--arch', '-a', metavar='ARCH', default='LightCNN')
parser.add_argument('--cuda', '-c', default=False)
parser.add_argument('--face_weight', default='weights/lightCNN_2_checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--emotion_weight', default='weights/mobilev3_large.pth', type=str, metavar='PATH',
                    help='')
parser.add_argument('--model', default='LightCNN-29v2', type=str, metavar='Model',
                    help='model type: LightCNN-9, LightCNN-29')
parser.add_argument('--root_path', default='res', type=str, metavar='PATH',
                    help='root path of face images (default: none).')
parser.add_argument('--save_path', default='', type=str, metavar='PATH', 
                    help='save root path for features of face images.')
parser.add_argument('--num_classes', default=7, type=int,
                    metavar='N', help='mini-batch size (default: 79077)')

def get_face_ID(configs, frame, instances, cur_frame_counter):
    global args
    args = parser.parse_args()

    if args.model == 'LightCNN-9':
        model = LightCNN_9Layers(num_classes=args.num_classes)
    elif args.model == 'LightCNN-29':
        model = LightCNN_29Layers(num_classes=args.num_classes)
    elif args.model == 'LightCNN-29v2':
        model = LightCNN_29Layers_v2(num_classes=args.num_classes)
    else:
        print('Error model type\n')
    model_emotion = mobilenetv3.mobilenetv3_large()
    model.eval()
    model_emotion.eval()
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
        model_emotion = model_emotion.cuda()

    if args.face_weight:
        if os.path.isfile(args.face_weight):
            # print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.face_weight , map_location='cpu')
            new_state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict) #checkpoint['state_dict']
    else:
        print("=> no checkpoint found at '{}'".format(args.face_weight))

    if args.emotion_weight:
        if os.path.isfile(args.emotion_weight):
            # print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint_emotion = torch.load(args.emotion_weight, map_location='cpu')
            model_emotion.load_state_dict(checkpoint_emotion)
    else:
        print("=> no checkpoint found at '{}'".format(args.emotion_weight))

    if len(instances) > 0:
        for instance in instances:
            ins = instance.get_latest_record()
            w = ins[2] - ins[1]
            h = ins[4] - ins[3]
            left = ins[1]-int(w/4)
            if left < 0 : left = 0
            right = ins[2]+int(w/4)
            top = ins[3]-int(h/8)
            if top < 0: top = 0
            bottom = ins[4]+int(h/8)

            transform = transforms.Compose([transforms.ToTensor()])
            img2 = frame[(top):(bottom), (left):(right), :]
            # cv2.imwrite('res/'+str(cur_frame_counter)+'.jpg', img2)

            '''人脸识别'''
            tag_face = ['Aa', 'Bb', 'Cc', 'Dd', 'Ee', 'Ff', 'Gg']

            img = np.dot(img2, [0.299, 0.587, 0.114])/255
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
            # img = np.reshape(img, ( 1,128, 128))
            # cv2.imwrite('res/' + str(10) + '.jpg', img)
            img = transform(img).unsqueeze(0)
            img = torch.tensor(img, dtype=torch.float32)

            # img = transform(img)
            # input_ = torch.zeros(1, 1, 128, 128)
            # input_[0, :, :, :] = img

            if args.cuda:
                img = img.cuda()
            _, features = model(img)
            face_id = tag_face[_.argmax()]

            if instance.his_face_id == face_id:  # 在人脸识别中连续两次识别为同一人，才是正确识别
                instance.face_id = face_id
            instance.his_face_id = face_id


            '''表情识别'''
            tag_emotion = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Anger', 'Disgust', 'Contempt']
            img2 = cv2.resize(img2, (224, 224), interpolation=cv2.INTER_CUBIC)
            # img2 = np.reshape(img2, (1,3, 224, 224))
            transform2 = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5], std=[0.5])])
            img2 = transform2(img2).unsqueeze(0)
            if args.cuda:
                img2 = img2.cuda()
            output = model_emotion(img2)
            emotion = tag_emotion[output.argmax()]

            if instance.his_emotion == emotion:  # 连续两个表情相同，才是正确的表情预测
                instance.emotion = emotion
            instance.his_emotion = emotion

            # print( instance.emotion)
            # save_feature(args.save_path, img_name, features.data.cpu().numpy()[0])



def save_feature(save_path, img_name, features):
    img_path = os.path.join(save_path, img_name)
    img_dir  = os.path.dirname(img_path) + '/';
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    fname = os.path.splitext(img_path)[0]
    fname = fname + '.feat'
    fid   = open(fname, 'wb')
    fid.write(features)
    fid.close()
