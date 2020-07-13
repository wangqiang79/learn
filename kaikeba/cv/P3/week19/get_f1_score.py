#coding:utf-8
import numpy as np
import shapely.geometry as plg
def evaluate_method(gt_pointList,det_pointList,IOU_CONSTRAINT):
    
    def polygon_from_points(points):
        """
        Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
        """        
        resBoxes=np.empty([1,8],dtype='int32')
        resBoxes[0,0]=int(points[0])
        resBoxes[0,4]=int(points[1])
        resBoxes[0,1]=int(points[2])
        resBoxes[0,5]=int(points[3])
        resBoxes[0,2]=int(points[4])
        resBoxes[0,6]=int(points[5])
        resBoxes[0,3]=int(points[6])
        resBoxes[0,7]=int(points[7])
        pointMat = resBoxes[0].reshape([2,4]).T
        return plg.Polygon(pointMat)    
    
    def rectangle_to_polygon(rect):
        resBoxes=np.empty([1,8],dtype='int32')
        resBoxes[0,0]=int(rect.xmin)
        resBoxes[0,4]=int(rect.ymax)
        resBoxes[0,1]=int(rect.xmin)
        resBoxes[0,5]=int(rect.ymin)
        resBoxes[0,2]=int(rect.xmax)
        resBoxes[0,6]=int(rect.ymin)
        resBoxes[0,3]=int(rect.xmax)
        resBoxes[0,7]=int(rect.ymax)

        pointMat = resBoxes[0].reshape([2,4]).T
        
        return plg.Polygon(pointMat)
    
    def rectangle_to_points(rect):
        points = [int(rect.xmin), int(rect.ymax), int(rect.xmax), int(rect.ymax), int(rect.xmax), int(rect.ymin), int(rect.xmin), int(rect.ymin)]
        return points
        
    def get_union(pD,pG):
        areaA = pD.area;
        areaB = pG.area;
        return areaA + areaB - get_intersection(pD, pG);
        
    def get_intersection_over_union(pD,pG):
        try:
            return get_intersection(pD, pG) / get_union(pD, pG);
        except:
            return 0
        
    def get_intersection(pD,pG):
        if not pD.intersects(pG):
            return 0        
        pInt = pD & pG
        return pInt.area
    
    
    # 标签和预测框匹配的总数
    matchedSum = 0
    # 所有样例的有效标签框总数
    numGlobalCareGt = 0;
    # 所有样例的有效预测框总数
    numGlobalCareDet = 0;
    

    # 对每张图片计算召回和准确度需要的量：预测正确的框数，gt中的框数，一共预测时框数
    for i in range(0,len(gt_pointList)):
        # 这张图片中预测框成功匹配的数量
        detMatched = 0
        
        # 预测框与标签框的IoU矩阵
        iouMat = np.empty([1,1])
        
        # 标签/预测多边形
        gtPols = []
        detPols = []
        
        # 标签/预测多边形的点集
        gtPolPoints = []
        detPolPoints = []
        
        # 标签框和预测框的匹配index对
        pairs = [] 
        # 所有成功匹配的预测框的index
        detMatchedNums = []
        
        # 该图片预测框的置信度
        arrSampleConfidences = [];
        # 该图片预测框是否与某个标签框匹配
        arrSampleMatch = [];

        
        # gt
        pointsList=gt_pointList[i]
	
        for n in range(len(pointsList)):
            points = pointsList[n]
            gtPol = polygon_from_points(points)
            # 记录标签多边形和点集
            gtPols.append(gtPol)
            gtPolPoints.append(points)
            
        #pre det 
	pointsList=det_pointList[i]
        for n in range(len(pointsList)):
            points = pointsList[n]
            # 将点坐标转化为多边形
            detPol = polygon_from_points(points)                    
            # 记录预测多边形和点集
            detPols.append(detPol)
            detPolPoints.append(points)
                            
        #用IoU计算gt与pre的匹配性 
        if len(gtPols)>0 and len(detPols)>0:
            #Calculate IoU and precision matrixs
            outputShape=[len(gtPols),len(detPols)]
            iouMat = np.empty(outputShape)
            # 匹配标记，0未匹配，1已匹配
            gtRectMat = np.zeros(len(gtPols),np.int8)
            detRectMat = np.zeros(len(detPols),np.int8)
            # 二重循环计算IoU矩阵
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum,detNum] = get_intersection_over_union(pD,pG)

            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    
                    # 若标签和预测框均为匹配，且均不忽略，则判断IoU
                    if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0:
                        # 若IoU大于某个阈值，则匹配成功
                        if iouMat[gtNum,detNum]>IOU_CONSTRAINT:
                            # 更新匹配标记
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            # 匹配数量+1
                            detMatched += 1
                            # 增加匹配对
                            pairs.append({'gt':gtNum,'det':detNum})
                            # 记录成功匹配的预测框index
                            detMatchedNums.append(detNum)

                            
        # 计算有效框的数量)
        numGtCare = len(gtPols)
        numDetCare = len(detPols)

        # 将该图片的计数记录到全局总数中
        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare
        
                                    
    # 计算全部图片的结果
    # 计算全部图片的召回率
    methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum)/numGlobalCareGt
    # 计算全部图片的准确率
    methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum)/numGlobalCareDet
    # 计算全部图片的hmean（即F-Score）
    methodHmean = 0 if methodRecall + methodPrecision==0 else 2* methodRecall * methodPrecision / (methodRecall + methodPrecision)
    
    methodMetrics = {'precision':methodPrecision, 'recall':methodRecall,'fscore/hmean': methodHmean}

    print methodMetrics


if __name__=='__main__':
    # gt_中有两张图片，每张图片上有一个检测框
    gt_pointLists=[
        [[100,100,200,100,200,200,100,200]],
	[[300,100,456,278,430,389,310,400]],
    ]
    # det 中也有两张图片，
    det_pointLists=[
        [[110,100,200,100,200,200,100,200],[300,100,456,278,430,389,310,400]],
	[[300,100,456,278,430,389,310,400]],
    ]
    IOU_CONSTRAINT=0.5    
    evaluate_method(gt_pointLists,det_pointLists,IOU_CONSTRAINT)
