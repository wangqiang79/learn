#coding:utf-8
import cv2
import numpy as np
from numpy.linalg import norm
import sys
import os
import json

SZ = 20          # size of training images
MAX_WIDTH = 1000 # maximum width of original image
MIN_AREA = 2000  # minimum area of license plate
PROVINCE_START = 1000

# 限制点的取值范围
def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0

def accurate_place(card_img_hsv, limit1, limit2, color,cfg):
    row_num, col_num = card_img_hsv.shape[:2]
    xl = col_num
    xr = 0
    yh = 0
    yl = row_num
    
    row_num_limit = cfg["row_num_limit"]
    col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5
    for i in range(row_num):
        count = 0
        for j in range(col_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if limit1 < H <= limit2 and 34 < S and 46 < V:
                count += 1
        if count > col_num_limit:
            if yl > i:
                yl = i
            if yh < i:
                yh = i
    for j in range(col_num):
        count = 0
        for i in range(row_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if limit1 < H <= limit2 and 34 < S and 46 < V:
                count += 1
        if count > row_num - row_num_limit:
            if xl > j:
                xl = j
            if xr < j:
                xr = j
    return xl, xr, yh, yl



def CaridDetect(car_pic):
    # 读取车牌图片到img中
    img = cv2.imread(car_pic)
    # 获取图片的大小
    pic_hight, pic_width = img.shape[:2]
    
    # 限定图片大小，凡是过宽，等比例缩小图片宽度到MAX_WIDTH
    resize_rate = 1
    if pic_width > MAX_WIDTH:
        resize_rate = MAX_WIDTH / pic_width
        img = cv2.resize(img, (MAX_WIDTH, int(pic_hight*resize_rate)), interpolation=cv2.INTER_AREA)
    
    # 打开配置文件,读取配置到cfg中
    f = open('config.js')
    j = json.load(f)
    for c in j["config"]:
        if c["open"]:
            cfg = c.copy()
            break
        else:
            raise RuntimeError('[ ERROR ] Invalid configuration.')

    # 如果配置中，设定blur大于零，则进行高斯模糊 
    blur = cfg["blur"]
    if blur > 0:
        img = cv2.GaussianBlur(img, (blur, blur), 0)

    oldimg = img
    # bgr 转 灰度图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    # 开运算后，求梯度图 
    kernel = np.ones((20, 20), np.uint8)
    img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0);

    # 二值化
    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Canny边缘检测
    img_edge = cv2.Canny(img_thresh, 100, 200)
   
    # 进行开闭运算，实现粘连物得分割
    kernel = np.ones((cfg["morphologyr"], cfg["morphologyc"]), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    # 利用opencv找连通区域
    try:
        contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 只取面积比MIN_AREA大的连通区域
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]
    
    car_contours = []
    for cnt in contours:
        # 取连通区域得最小外接矩形
	# 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        rect = cv2.minAreaRect(cnt)
	
	# 求矩形长短边之比wh_ratio
        area_width, area_height = rect[1]
	#  这里是对竖向车牌的适应，但是下面没有看到旋转代码？
        if area_width < area_height:
            area_width, area_height = area_height, area_width
        wh_ratio = area_width / area_height
        
	#长短边之比介于2--5.5之间，才是车牌得长短边之比
        if wh_ratio > 2 and wh_ratio < 5.5:
            car_contours.append(rect)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

    card_imgs = []
    bboxes = []
    import pdb
    pdb.set_trace()
    for rect in car_contours:
        if rect[2] > -1 and rect[2] < 1:
            angle = 1
        else:
            angle = rect[2]
	# 中心点，宽，高，角度
        rect = (rect[0], (rect[1][0]+5, rect[1][1]+5), angle)

        box = cv2.boxPoints(rect)
        heigth_point = right_point = [0, 0]
        left_point = low_point = [pic_width, pic_hight]
	# 将外接矩形的四个点分为左点，右点，上点，下点
        for point in box:
            if left_point[0] > point[0]:
                left_point = point
            if low_point[1] > point[1]:
                low_point = point
            if heigth_point[1] < point[1]:
                heigth_point = point
            if right_point[0] < point[0]:
                right_point = point
        bboxes.append([left_point, right_point, low_point, heigth_point])
        import pdb
        pdb.set_trace()
	# 拉伸变换，将车牌摆正
        if left_point[1] <= right_point[1]:
            new_right_point = [right_point[0], heigth_point[1]]
            pts2 = np.float32([left_point, heigth_point, new_right_point])
            pts1 = np.float32([left_point, heigth_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
            point_limit(new_right_point)
            point_limit(heigth_point)
            point_limit(left_point)
            card_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
            card_imgs.append(card_img)
            
        elif left_point[1] > right_point[1]:
            
            new_left_point = [left_point[0], heigth_point[1]]
            pts2 = np.float32([new_left_point, heigth_point, right_point])
            pts1 = np.float32([left_point, heigth_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
            point_limit(right_point)
            point_limit(heigth_point)
            point_limit(new_left_point)
            card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
            card_imgs.append(card_img)
            
    # 车牌颜色识别，利用颜色再对车片区域进行优化
    # 颜色识别主要得思路是把车牌图片由RGB转为HSV格式，再分别判断H,S，V。 3个通道到的取值范围来确定车牌得颜色。关于HSV格式，我又篇博文可以参考下：
    # https://zhuanlan.zhihu.com/p/129314401
    colors = []
    for card_index,card_img in enumerate(card_imgs):
        green = yello = blue = black = white = 0
        if card_img is None or card_img.shape[0] == 0 or card_img.shape[1] == 0:
            continue
        # BGR图片转为HSV
        card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
        
        if card_img_hsv is None:
            continue
        row_num, col_num= card_img_hsv.shape[:2]
        card_img_count = row_num * col_num

        for i in range(row_num):
            for j in range(col_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                # 黄色：H通道 加介于11-34,S通道的值大于34
                if 11 < H <= 34 and S > 34:
                    yello += 1
                # 绿色：H介于35-99之间，S大于34
                elif 35 < H <= 99 and S > 34:
                    green += 1
                # 蓝色：H介于99-124之间，
                elif 99 < H <= 124 and S > 34:
                    blue += 1
                # 黑色：V介于0-46
                if 0 < H <180 and 0 < S < 255 and 0 < V < 46:
                    black += 1
                # 白色：S 介于0-34,V介于221-225
                elif 0 < H <180 and 0 < S < 43 and 221 < V < 225:
                    white += 1
        color = "no"

        limit1 = limit2 = 0
        if yello*2 >= card_img_count:
            color = "yello"
            limit1 = 11
            limit2 = 34
        elif green*2 >= card_img_count:
            color = "green"
            limit1 = 35
            limit2 = 99
        elif blue*2 >= card_img_count:
            color = "blue"
            limit1 = 100
            limit2 = 124
        elif black + white >= card_img_count*0.7:
            color = "bw"
            
        colors.append(color)
        
        if limit1 == 0:
            continue
            
        xl, xr, yh, yl = accurate_place(card_img_hsv, limit1, limit2, color,cfg)
        if yl == yh and xl == xr:
            continue
        need_accurate = False
        if yl >= yh:
            yl = 0
            yh = row_num
            need_accurate = True
        if xl >= xr:
            xl = 0
            xr = col_num
            need_accurate = True
        card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh-yl)//4 else card_img[yl-(yh-yl)//4:yh, xl:xr]
        if need_accurate:
            card_img = card_imgs[card_index]
            card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            xl, xr, yh, yl = accurate_place(card_img_hsv, limit1, limit2, color,cfg)
            if yl == yh and xl == xr:
                continue
            if yl >= yh:
                yl = 0
                yh = row_num
            if xl >= xr:
                xl = 0
                xr = col_num
 
        card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh-yl)//4 else card_img[yl-(yh-yl)//4:yh, xl:xr]
        # 车牌图片
        roi = card_img
	# 车牌对颜色
        card_color = color
        left_point, right_point, low_point, heigth_point = bboxes[card_index]
	# 车牌得四个顶点
        labels = (int(left_point[0]/resize_rate), int(right_point[0]/resize_rate), \
                  int(low_point[1]/resize_rate), int(heigth_point[1]/resize_rate))

            
    return roi, labels, card_color
