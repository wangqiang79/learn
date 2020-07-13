import torch
import torch.nn as nn


def get_class_balanced_cross_enptry(gt_score,pred_score):
	#请写出类平衡交叉熵的liss
	return score_loss


def get_geo_loss(gt_geo, pred_geo):
	#写出d1,d2,d3,d4,4个feature map的iou_loss 和 angle_map的loss
	return iou_loss, angle_loss


class Loss(nn.Module):
	def __init__(self, weight_angle=10):
		super(Loss, self).__init__()
		self.weight_angle = weight_angle

	def forward(self, gt_score, pred_score, gt_geo, pred_geo, ignored_map):
		if torch.sum(gt_score) < 1:
			return torch.sum(pred_score + pred_geo) * 0
		
		classify_loss = get_class_balanced_cross_enptry(gt_score, pred_score*(1-ignored_map))
		iou_loss, angle_loss = get_geo_loss(gt_geo, pred_geo)

		geo_loss = self.weight_angle * angle_loss + iou_loss
		print('classify loss is {:.8f}, angle loss is {:.8f}, iou loss is {:.8f}'.format(classify_loss, angle_loss, iou_loss))
		return geo_loss + classify_loss
