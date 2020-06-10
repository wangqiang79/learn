import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp
from data import cfg

import numpy as np

GPU = cfg['gpu_train']

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        # num_classes=2
        # overlap_thresh=0.35
        # prior_for_matching=True
        # bkg_label=0
        # neg_mining=True
        # neg_pos=7
        # neg_overlap=0.35
        # encode_target=False
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        # print('predictions: ', predictions)
        loc_data, conf_data = predictions   # loc_data shape: torch.Size([64, 21824, 4])
                                            # conf_data shape: torch.Size([64, 21824, 2])
        # print(conf_data)
        # print(conf_data.shape)
        priors = priors                     # priors shape: torch.Size([21824, 4])
        # priors: tensor([[0.x, 0.x, 0.x, 0.x], [0.x, 0.x, 0.x, 0.x], [0.x, 0.x, 0.x, 0.x], ...], device='cuda:0')
        # print(priors)
        # print('priors shape: ', priors.shape)
        num = loc_data.size(0)              # num: 64, this is batch size
        # print('num: ', num)
        num_priors = (priors.size(0))       # num_priors: 21824, total number of anchors
        # print('num_priors: ', num_priors)   # num_priors: bigger: 21824, smaller: 5440

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)        # loc_t: torch.Size([64, 21824, 4])
        # print('loc_t: ', loc_t.shape)
        conf_t = torch.LongTensor(num, num_priors)      # conf_t: torch.Size([64, 21824])
        # conf_t: tensor([[0 ,0 ...], [0 ,0, 1, ...], ...])
        # print(conf_t)
        # print('conf_t: ', conf_t.shape)

        # print('target shape: ', np.array(targets).shape)    # targets shape: 64,
        # print('targets[0]: ', targets[0])
        # targets[0]: tensor([[0.x, 0.x, 0.x, 0.x, 1.], [0.x, 0.x, 0.x, 0.x, 1.], ...], device='cuda:0')
        # targets[0]: tensor([[0.x, 0.x, 0.x, 0.x ,1.]], device='cuda:0')
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            # truths: tensor([[0.6023, 0.6023, 0.7455, 0.7705], [0.0000, 0.2205, 0.1795, 0.3864]], device='cuda:0')
            # labels: tensor([1., 1.], device='cuda:0')
            # defaults: tensor([[], [], [], ...], device='cuda:0')

            # print('idx: ', idx)
            # print('truths: ', truths)
            # print('labels: ', labels)
            # print('defaults: ', defaults)

            # threshold: 0.35
            # variance: [0.1, 0.2]
            # idx: 0 or 1 or ...or 63, which image
            # loc_t: [64, 21824, 4]
            # conf_t: [64, 21824]    loc/conf_t: prior boxes
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        if GPU:
            loc_t = loc_t.cuda()        # offset to train: cxcy & wh
            conf_t = conf_t.cuda()

        # conf[best_truth_overlap < threshold] = 0,
        #                 dim = 21824, which is also the prior number
        # conf_t: tensor([[0, 0, ...],      num=64
        #                 [0, 0, ...],
        #                 ...])
        # conf_t.shape: torch.Size([64, 21824])
        #
        # loct_t                      torch.Size([64, 21824, 4])
        pos = conf_t > 0            # torch.Size(64, 21824)
        # pos: tensor([[False, False, ...],      num=64
        #              [False, False, ...],      almost all false
        #              ...])
        # print(pos)
        # print(loc_t.shape)

        # Localization Loss (Smooth L1)
        '''pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data) explanation'''
        # Shape: [batch,num_priors,4]
        """ pos.dim() = 2 """
        # print(pos.unsqueeze(1).shape)   # ([64, 1, 21824])
        #   ([[[False, False, ...]],
        #     [[False, False, ...]]
        #     ...])
        #
        """ pos.unsqueeze(2)                # [64, 21824, 1] """
        # ([[[False], [False], [], ...],
        #   [[False], [False], [], ...],
        #   ...
        #   ])
        # expand_as: Expand this tensor to the same size as other.self.expand_as(other)
        #            is equivalent to self.expand(other.size())
        # expand e.g:
        # x = torch.tensor([[1], [2], [3]])
        # x.size() = ([3, 1])
        # x.expand(3, 4)
        # x = tensor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
        # x.expand(-1, 4) = x
        # -1 means not changing the size of that dimension
        """ here, loc_data = torch.Size([64, 21824, 4]) """
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)      # torch.Size([64, 21824, 4])
        # pos_idx: tensor([[[False, False, False, False], [], ...], [[], [], ...[]], ...])
        # print(pos_idx.shape)
        loc_p = loc_data[pos_idx].view(-1, 4)
        # loc_p: positive predicted sample(prior)s location, tensor([[1.074, -0.836, -0.934, 0.414], [x, x, x, x], ...])
        # loc_p.shape: torch.Size([1186, 4]), torch.Size([num of True, 4])
        loc_t = loc_t[pos_idx].view(-1, 4)
        # loc_t: positive sample(prior)s location, tensor([[1.0743, -0.8366, -0.9314, 0.4146], [x, x, x, x], ...])
        # loc_t.shape: torch.Size([1186, 4]), torch.Size([num of True, 4])
        ''' _t & _p'''
        # loc_p: predicted results of offsets between predicted bboxes and prior boxes
        # loc_t: target of offsets between ground truth bboxes and prior boxes
        # because the prior boxes are fixed, so we need to to use the same indices (pos_idx)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        # loss_l: just give out a really big number initially. 5031.2363, 4719.4766, 1720.2607, ...

        ''' now we are dueling with classes '''
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # conf_data.shape: torch.Size([64, 21824, 2])
        # batch_conf.shape: torch.Size(64x21824=1396736, 2)
        # batch_conf:
        # tensor([[0.0473, -0.1172], [0.1001, 0.2789], ...])
        # conf_t.shape: torch.Size([64, 21824]),
        # conf_t: almost all 0
        #
        # log_sum_exp: log(softmax(batch_conf))
        # log_sum_exp - batch_conf(0 (background) or 1 (face) is determined by prior label(conf_t))
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        # log_sum_exp(.).shape = batch_conf.shape = torch.Size([64x21824=1396736, 1])
        # print(log_sum_exp(batch_conf).shape)
        # print(batch_conf.gather(1, conf_t.view(-1, 1)).shape)

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        # pos = conf_t > 0  # torch.Size(64, 21824)
        # pos: tensor([[False, False, ...],      num=64
        #              [False, False, ...],      almost all false
        #              ...])

        loss_c = loss_c.view(num, -1)       # torch.Size([64, 21824])
        _, loss_idx = loss_c.sort(1, descending=True)   # sort the loss for each image in a batch
        # loss_idx: tensor([[6264, 4904, ....], [], ...])      torch.Size([64, 21824])
        # _: is sorted matrix
        _, idx_rank = loss_idx.sort(1)
        # sort the loss_idx under ascending order
        # idx_rank,     torch.Size([64, 21824])
        # print(_)
        # print(idx_rank)
        # print(idx_rank.shape)
        num_pos = pos.long().sum(1, keepdim=True)
        # num_pos: tensor([[13] ,[10] , ...])    torch.Size([64, 1])
        # num_pos: for each image, how many positive samples
        # print(num_pos)
        # print(num_pos.shape)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        # clamp: clamp all elements in input into the range [min, max]
        # self.negpos_ratio*num_pos: 7 * each element in num_pos,   torch.Size([64, 1])
        # tensor([[49], [105], [70], ...])
        # num_neg: elements cannot be over the number of priors, which is pos.size(1) - 1 = 21823 here
        ''' Hard negative mining '''
        ''' Get samples with num_neg smallest losses '''
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # eg1.
        # indices: 0  1  2  3  4  5  6
        # idx:     6  5  4  3  2  1  0 -> loss descending
        # rank:    6  5  4  3  2  1  0 -> idx ascending
        # num_neg = 2
        # so 1 & 0 in rank is picked
        # 1 & 0 in rank corresponds to 5 & 6 in indices,
        # which also corresponds to the first two (6 & 5) in idx which follows the loss descending order.
        # so we pick the samples with smallest losses. Done!
        #
        # eg2.
        # indices: 0  1  2  3  4
        # idx:     3  4  0  2  1
        # rank:    2  4  3  0  1
        # num_neg = 3
        # so 2, 0, 1 in rank is picked
        # which corresponds to 0 3 4 in indices,
        # which corresponds to the first three (3, 4 & 0) in idx wich follows the loss descending order.
        # so we pick the samples with smallest losses. Done!
        #
        # neg.shape: torch.Size([64, 21824])
        # neg: tensor([[False, False, ...],      num=64
        #              [False, False, ...],      almost all false
        #              ...])

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # pos/neg.unsqueeze(2):
        #   tensor([[[False], [False], ...], [[False], [False], ...], ...])
        #   shape: torch.Size([64, 21824, 1])
        # pos_idx/neg_idx:
        #   tensor([[[False], [False], ...], [[False], [True], ...], ...])
        #   shape: torch.Size([64, 21824, 2])
        # print(pos_idx.shape)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1,self.num_classes)     # this is the target
        # pos_idx + neg_idx: like "or" operation
        # .gt(0): if (pos_idx + neg_idx) > 0, then that element is True, otherwise False.
        #
        # conf_data[(pos_idx + neg_idx).gt(0)]: grab the loss where there losses are True.
        # conf_data[·].shape: torch.Size([18080])
        # conf_data[·]: tensor([-2.5210, 1.1987, -2.3412, ...])
        targets_weighted = conf_t[(pos+neg).gt(0)]  # this is the ground truth
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c
