手写数字识别代码 
================  
## 导航目录
 - [仓库说明](#仓库说明)
 - [克隆仓库](#克隆仓库)
 - [代码提交](#仓库说明)
 - [文件目录结构](#文件目录结构)
 - [week17作业](#week17作业)
 - [参与贡献](#参与贡献)

仓库说明  
--------  

 一份手写字符识别的代码，含一个手写字符识别器，一个手写板。  

克隆仓库  
--------  

```
git clone https://gitee.com/anjiang2020_admin/formula_recognize_20200620/tree/master/week17/hand_number_recog
```

代码提交  
--------  

```  
git add .
git commit -m "your content"
git push -u origin master
```  

文件目录结构  
--------  
 - 用于训练手写字符的数据集：MNIST_data
 - 获取手写字符图片的手写板：get_handwrite.py
 - 保存手写字符文件夹：hand_write
 - 手写板背景：hand_write_src.png
 - 手写板背景：hand_write_src_b.png
 - 识别程序：inference3.py
 - kill_hw
 - 数据处理程序：mnist
 - readme.md
 - 模型保存文件夹：save
 - 模型训练：train.py
 - write_image.py


'''
运行办法：
  - python get_handwrite.py 如下下图。然后自己手写一个数字,这个数字的图片会自动保存到：./hand_write/text.png
![输入图片说明](https://images.gitee.com/uploads/images/2020/0706/134343_d188efab_7401441.png "屏幕截图.png")
  - python inderence3.py 对上步写的手写数字./hand_write/text.png进行识别。
![输入图片说明](https://images.gitee.com/uploads/images/2020/0706/133938_044dd88b_7401441.png "屏幕截图.png")
  - python train.py 利用手写数字数据集mnist对训练一个手写数字的模型。

'''

参与贡献  
--------  

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request

码云特技  
--------  

1. 使用 README\_XX.md 来支持不同的语言，例如 README\_en.md, README\_zh.md  
2. 码云官方博客 [blog.gitee.com](https://blog.gitee.com)  
3. 你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解码云上的优秀开源项目  
4. [GVP](https://gitee.com/gvp) 全称是码云最有价值开源项目，是码云综合评定出的优秀开源项目  
5. 码云官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)  
6. 码云封面人物是一档用来展示码云会员风：采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)  

