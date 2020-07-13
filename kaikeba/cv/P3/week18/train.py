#coding:utf-8
import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset import custom_dataset
from model import EAST
from loss import Loss
import os
import time
import numpy as np


def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval):
	# 数据处理 
	#import pdb
	#pdb.set_trace()
	file_num = len(os.listdir(train_img_path))
	trainset = custom_dataset(train_img_path, train_gt_path)
	train_loader = data.DataLoader(trainset, batch_size=batch_size, \
                                   shuffle=True, num_workers=num_workers, drop_last=True)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	#模型实现
	model = EAST()
	data_parallel = False
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		data_parallel = True
	model.to(device)
	
	#loss实现
	criterion = Loss()

	#[完善优化算法的调用]写出优化算法的
	optimizer = torch.optim.


	#定义学习策略
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter//2], gamma=0.1)

	for epoch in range(epoch_iter):	
		model.train()
		epoch_loss = 0
		epoch_time = time.time()
		for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
			start_time = time.time()
			#import pdb
			#pdb.set_trace()
			print("start_time=%s"%(start_time))
			img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
			
			# 使用模型
			pred_score, pred_geo = model(img)
			# 计算得到loss
			loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
			
			epoch_loss += loss.item()

			# 利用loss求取梯度
			optimizer.zero_grad()
			loss.backward()

			#权重更新
			optimizer.step()
			scheduler.step()
			print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
              epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))
		
		print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time))
		print(time.asctime(time.localtime(time.time())))
		print('='*50)
		if (epoch + 1) % interval == 0:
			state_dict = model.module.state_dict() if data_parallel else model.state_dict()
			torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch+1)))


if __name__ == '__main__':
	train_img_path = os.path.abspath('./ICDAR_2015/train_img')
	train_gt_path  = os.path.abspath('./ICDAR_2015/train_gt')
	pths_path      = './pths'
	#batch_size     = 24 
	batch_size     = 1 
	lr             = 1e-3
	#num_workers    = 4
	num_workers    = 1
	epoch_iter     = 600
	save_interval  = 5
	train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval)
	
