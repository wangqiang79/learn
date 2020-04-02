#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Pw @ 2020-02-05 22:38:54

import json
import os
import random
from collections import Counter

karpathy_json_path = "../../caption_datasets/dataset_coco.json"

#file_path = "./"
file_path = "../../mscoco/val2017/"
all_img_files = []
for files in os.walk(file_path):
    all_img_files = files[2]

with open(karpathy_json_path, 'r') as j:
    data = json.load(j)

word_freq = Counter()
max_len = 100
image_folder = './'

paths = []
new_data = {}
new_imgs = []
 
total = 0
deno = 0
for img in data['images']:
    captions = []
    deno += 1
    total += len(img['sentences'])
    for c in img['sentences']:
        # Update word frequency
        word_freq.update(c['tokens'])
        if len(c['tokens']) <= max_len:
            captions.append(c['tokens'])

    if len(captions) == 0:
        continue

    path = os.path.join(image_folder, img['filepath'], img['filename'])
    sp = img['filename'].split("_")
    index = sp[-1]
    if index in all_img_files:
        img['filepath'] = 'val2017'
        img['filename'] = index
        r = random.random()
        if r >0.2:
            img['split'] = 'train'
        if r <= 0.2 and r >0.1:
            img['split'] = 'val'
        if r <=0.1:
            img['split'] = 'test'
        new_imgs.append(img)
new_data['images'] = new_imgs
print(float(total)/float(deno))
out = json.dumps(new_data)
with open('new_data.json', 'w', encoding='utf-8') as f:
    f.write(out)
