import cv2
import numpy as np
from video_helper import VideoHelper
from config import Configs
import math

# predict的时候预测有6个值：xc,yc,vx,vy,w,h
# observation 的时候只有4个值：xc,yc,w,h

class KalmanFilter(object):
    def __init__(self,video_helper):
        self.dynamParamsSize = 6
        self.measureParamsSize = 4
        self.kalman = cv2.KalmanFilter(dynamParams=self.dynamParamsSize,
                                       measureParams = self.measureParamsSize)
        self.first_run = True
        dT = 1./video_helper.frame_fps
        # 转移矩阵 Transition matrix
        self.kalman.transitionMatrix = np.array([[1,0,dT,0,0,0],
                                                 [0,1,0,dT,0,0],
                                                 [0,0,1,0,0,0],
                                                 [0,0,0,1,0,0],
                                                 [0,0,0,0,1,0],
                                                 [0,0,0,0,0,1]],np.float32)

        self.kalman.measurementMatrix = np.array([[1,0,0,0,0,0],
                                                  [0,1,0,0,0,0],
                                                  [0,0,0,0,1,0],
                                                  [0,0,0,0,0,1]],np.float32)


        self.kalman.processNoiseCov = np.array([[0.01,0,0,0,0,0],
                                                [0,0.01,0,0,0,0],
                                                [0,0,5.0,0,0,0,],
                                                [0,0,0,5.0,0,0],
                                                [0,0,0,0,0.01,0],
                                                [0,0,0,0,0,0.01]],np.float32)
        self.kalman.measurementNoiseCov = np.array([[0.1,0,0,0],
                                                    [0,0.1,0,0],
                                                    [0,0,0.1,0]
                                                    [0,0,0,0.1]].np.float32)
    # 单纯依靠kalman filter给出bbox
    def get_predicted_bbox(self):

        predicted_res = self.kalman.predict().T(0)
        #需要改成自己的格式
        predicted_bbox = self.get_bbox_from_kalman_form(predicted_res)
        return predicted_bbox

    # 对kalman filter进行的predict进行correct
    #后面两个方程的内容
    def corret(self,bbox):

        # bbox: left,right,top,bottom
        w = bbox[1] - bbox[0] + 1  #
        h = bbox[3] - bbox[2] + 1
        xc = int(bbox[0] + w/2.)
        yc = int(bbox[2] + h/2.)

        measurement = np.array([[xc,yc,w,h]],dtype = np.float32)

        if self.first_run:

            self.kalman.statePre = np.array([measurement[0],measurement[1],
                                             [0],[0],
                                            measurement[2],measurement[3]],dtype=np.float32)
            self.first_run = False
        corrected_res = self.kalman.correct(measurement).T[0]
        corrected_bbox = self.get_bbox_from_kalman_form(corrected_res)
        return corrected_bbox

    def get_bbox_from_kalman_form(self,kalman_form):
        xc = kalman_form[0]
        yc = kalman_form[1]
        w = kalman_form[4]
        h = kalman_form[5]
        l = math.ceil(xc - w/2.)
        r = math.ceil(xc + w/2.)
        t = math.ceil(yc -h/2.)
        b = math.ceil(yc + h/2.)
        return [l,r,t,b]









