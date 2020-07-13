#coding:utf-8
import fitz

'''
# 将PDF转化为图片
pdfPath pdf文件的路径
imgPath 图像要保存的文件夹
zoom_x x方向的缩放系数
zoom_y y方向的缩放系数
rotation_angle 旋转角度
'''
def pdf_image(pdfPath,imgPath,zoom_x,zoom_y,rotation_angle):
    # 打开PDF文件
    pdf = fitz.open(pdfPath)
    # 逐页读取PDF
    for pg in range(0, pdf.pageCount):
        page = pdf[pg]
        # 设置缩放和旋转系数
        trans = fitz.Matrix(zoom_x, zoom_y).preRotate(rotation_angle)
        pm = page.getPixmap(matrix=trans, alpha=False)
        # 开始写图像
        pm.writePNG(imgPath+str(pg)+".png")
    pdf.close()
    
#pdf_image(r"C:\Users\12624\Desktop\a.pdf",r"C:\Users\12624\Desktop\\",5,5,0)
pdfPath="../../CVPR2020论文合集/xMUDA_\ Cross-Modal\ Unsupervised\ Domain\ Adaptation\ for\ 3D\ Semantic\ Segmentation.pdf "
imgPath="./imgFromPdf"
pdf_image(pdfPath,imgPath,5,5,0)
