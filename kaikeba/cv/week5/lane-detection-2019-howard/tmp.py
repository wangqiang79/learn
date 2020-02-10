import cv2
import numpy as np
from PIL import Image
from matplotlib import image

# img = cv2.imread('/wd-howard/dataset/lane_detection/baidu/train/label_fixed/Label_road03/Label/Record001/Camera 5/171206_025753822_Camera_5_bin.png', cv2.IMREAD_COLOR)
# img = cv2.imread('/wd-howard/dataset/lane_detection/baidu/train/label_fixed/Label_road03/Label/Record001/Camera 5/171206_025753822_Camera_5_bin.png', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('/wd-howard/dataset/lane_detection/baidu/train/Gray_Label/Label_road03/Label/Record001/Camera 5/171206_025753822_Camera_5_bin.png', cv2.IMREAD_GRAYSCALE)
# img = Image.open('/wd-howard/dataset/lane_detection/baidu/train/Gray_Label/Label_road03/Label/Record001/Camera 5/171206_025753822_Camera_5_bin.png')
# img = Image.open('/wd-howard/dataset/lane_detection/baidu/train/label_fixed/Label_roajjkjd03/Label/Record001/Camera 5/171206_025753822_Camera_5_bin.png')
# img = np.array(img)
print(img.shape)
# indices = np.where(img[1000, :] > 0)[0]
# print(indices)
# print(img[1000, 1405:1420])
print(img[1000, 1405])