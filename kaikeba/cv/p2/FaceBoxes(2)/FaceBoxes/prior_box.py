import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        #self.aspect_ratios = cfg['aspect_ratios']
        self.min_sizes = cfg['min_sizes']   # [[32, 64, 128], [256], [512]]
        self.steps = cfg['steps']           # [32, 64, 128]
        self.clip = cfg['clip']             # False
        self.image_size = image_size        # 1024
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.we_use_small_model = cfg['we_use_small_model'] # we need't that big
        # [[32, 32], [16, 16], [8, 8]]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            # k = 0, f = [32, 32]
            # k = 1, f = [16, 16]
            # k = 2, f = [8, 8]
            min_sizes = self.min_sizes[k]
            # k = 0: min_sizes = [32, 64, 128]
            # k = 1: min_sizes = [256]
            # k = 2: min_sizes = [512]
            for i, j in product(range(f[0]), range(f[1])):  # product([0,31], [0,31])
                # product: cartesian product
                # i: 0, j: 0
                # i: 0, j: 1
                # ...
                # i: 7: j: 7
                # (...)
                # (i: 31, j: 31)
                if self.we_use_small_model:
                    for min_size in min_sizes:
                        s_kx = min_size / self.image_size[1]    # (32,    64,    128,  256) / 512
                        s_ky = min_size / self.image_size[0]    # 0.0625, 0.125, 0.25, 0.5
                        if min_size == 32:
                            dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.5]]
                            dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.5]]
                            for cy, cx in product(dense_cy, dense_cx):
                                anchors += [cx, cy, s_kx, s_ky]
                        # elif min_size == 64:
                        #     dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.5]]
                        #     dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.5]]
                        #     for cy, cx in product(dense_cy, dense_cx):
                        #         anchors += [cx, cy, s_kx, s_ky]
                        else:
                            cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                            cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                            anchors += [cx, cy, s_kx, s_ky]
                else:
                    for min_size in min_sizes:
                        s_kx = min_size / self.image_size[1]    # (32,      64,     128,   256,  512) / 1024
                        s_ky = min_size / self.image_size[0]    # 0.03125, 0.0625, 0.125, 0.25, 0.5
                        if min_size == 32:
                            dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.25, j+0.5, j+0.75]]
                            dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.25, i+0.5, i+0.75]]
                            for cy, cx in product(dense_cy, dense_cx):
                                anchors += [cx, cy, s_kx, s_ky]
                        elif min_size == 64:
                            dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.5]]
                            dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.5]]
                            for cy, cx in product(dense_cy, dense_cx):
                                anchors += [cx, cy, s_kx, s_ky]
                        else:
                            cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                            cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                            anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
