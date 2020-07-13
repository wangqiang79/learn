import torch
#数据增强处理包
from torch.utils import data
# 神经网络包
from torch import nn
#引入学习率变化策略包
from torch.optim import lr_scheduler
#引入我们自己写的数据加载包，把数据整理成pytorch需要的格式
from dataset import custom_dataset
#引入我们自己写得EAST模型
from model import EAST
#引入我们自己写得Loss
from loss import Loss
import os
import time
import numpy as np


def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval):
        # 得到文件夹下得文件列表，然后计算数目
	file_num = len(os.listdir(train_img_path))
	#利用我们自己custom_dataset把数据从文件夹里读取，并整理成pytorch.utils.data.DataLoader需要的格式trainset
	trainset = custom_dataset(train_img_path, train_gt_path)
	#把数据按照batch_size,shuffle,num_workers,drop_last等原则进行整理
	train_loader = data.DataLoader(trainset, batch_size=batch_size, \
                                   shuffle=True, num_workers=num_workers, drop_last=True)
        #加载我们自己实现的loss函数	
	criterion = Loss()
	#获取计算设备，使用gpu 或者 cpu
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# 加载我们自己实现的模型
	model = EAST()
	#是否并行计算
	data_parallel = False
	# 如果有gpu设备，就把模型转为并行模型，方便用并行得方式把数据输入到模型中
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		data_parallel = True
        #把模型绑定到设备上（gpu or cpu）上
	model.to(device)
        
	#优化器，把模型得参数给到优化器，让优化器按照lr去更新这些参数
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	#设定优化器工作的时候，lr的更新策略
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter//2], gamma=0.1)

	for epoch in range(epoch_iter):	
                # 函数train和函数eval的作用是将Module及其SubModule分别设置为training mode和evaluation mode。这两个函数只对特定的Module有影响，例如Class Dropout、Class BatchNorm	        
		model.train()
		# 每个epoch开始前，更新一个学习率
		scheduler.step()
		epoch_loss = 0
		epoch_time = time.time()
		for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
			start_time = time.time()
			img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
			# 将图片输入east模型中，得到score map 和 geo_map
			pred_score, pred_geo = model(img)
			# 计算score的loss,geomerty的loss
			loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
		        # epoch内累加loss	
			epoch_loss += loss.item()
			# 原来梯度清零
			optimizer.zero_grad()
			# 重新计算梯度
			loss.backward()
			# 应用更新公式
			optimizer.step()
                        #打印当前epoch,当前iter的mini-batch,time cost,batch_loss等信息
			print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
              epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))
		#打印整个epoch的loss,时间耗费
		print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time))
		print(time.asctime(time.localtime(time.time())))
		print('='*50)
		# 保存模型参数
		if (epoch + 1) % interval == 0:
			state_dict = model.module.state_dict() if data_parallel else model.state_dict()
			torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch+1)))


if __name__ == '__main__':
        #图片路径
	train_img_path = os.path.abspath('../ICDAR_2015/train_img')
	#图片标注路径
	train_gt_path  = os.path.abspath('../ICDAR_2015/train_gt')
	#模型保存路径,含预先训练得模型和本次训练得模型
	pths_path      = './pths'
	# 训练得时候得batch_size
	#batch_size     = 24 
	batch_size     = 1 
	# 初始学习旅
	lr             = 1e-3
	#数据处理得线程书目
	num_workers    = 1
	#训练600代
	epoch_iter     = 600
	#每5秒钟，出发保存一次中间模型
	save_interval  = 5
	#
	train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval)
	
