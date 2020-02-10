"""
@description: 执行推断
"""


"""
import
"""
from config import ConfigInference
import utils
from os.path import join as pjoin
import pandas as pd
import numpy as np
import cv2
import torch
import time


"""
main
"""
if __name__ == '__main__':
    cfg = ConfigInference()
    print('Pick device: ', cfg.DEVICE)
    device = torch.device(cfg.DEVICE)

    # 网络
    print('Generating net: ', cfg.NET_NAME)
    net = utils.create_net(3, cfg.NUM_CLASSES, net_name=cfg.NET_NAME)
    net.eval()

    # 加载预训练权重
    print('Load pretrain weights: ', cfg.PRETRAINED_WEIGHTS)
    net.load_state_dict(torch.load(cfg.PRETRAINED_WEIGHTS, map_location='cpu'))
    net.to(device)

    # 数据生成器
    print('Preparing data... batch_size: {}, image_size: {}, crop_offset: {}'.format(cfg.BATCH_SIZE, cfg.IMAGE_SIZE, cfg.HEIGHT_CROP_OFFSET))
    # todo
    data_generator = utils.test_data_generator(cfg.IMAGE_ROOT,
                                                cfg.BATCH_SIZE, cfg.IMAGE_SIZE, cfg.HEIGHT_CROP_OFFSET)

    # 推断
    print('Let us inference ...')
    done_num = 0
    while True:
        images, images_filename = next(data_generator)
        if images is None:
            break
        images = images.to(device)
        
        predicts = net(images)
        predicts = predicts.cpu().detach().numpy()
        
        # 恢复成原先的尺寸
        outs = utils.decodePredicts(predicts, cfg.IMAGE_SIZE_ORG, cfg.HEIGHT_CROP_OFFSET, mode='color')

        # 保存
        for i, out in enumerate(outs):
            cv2.imwrite(pjoin(cfg.LABEL_ROOT, images_filename[i].replace('.jpg', '_bin.png')), out)
            org_image = cv2.imread(pjoin(cfg.IMAGE_ROOT, images_filename[i]))
            overlay_image = cv2.addWeighted(org_image, 0.6, out, 0.4, gamma=0)
            cv2.imwrite(pjoin(cfg.OVERLAY_ROOT, images_filename[i].replace('.jpg', '.png')), overlay_image)

        done_num += len(images_filename)
        print('Finished {} images'.format(done_num))

    print('Done')
