# FR-Loss
人脸识别损失函数对比总结
参考知乎链接：
https://zhuanlan.zhihu.com/p/34404607<br>https://zhuanlan.zhihu.com/p/34436551
# 以下实验基于mnist手写体数据集
## Softmax 
softmax只保证能够正确的分类，但是无法保证同一类特征之间类内紧致，类间间隔大，如下图：
![image](https://github.com/LrpljL/FR-Loss/blob/master/Raw/cos_epoch%3D79.jpg)
![image](https://github.com/LrpljL/FR-Loss/blob/master/Raw/epoch%3D79.jpg)
## Triple loss
以三元组（a, p, n）形式进行优化，不同类特征的L2距离要比同类特征的L2距离大margin m，同时获得类内紧凑和类间分离，其中三元组的选择极具技巧性。
![image](https://pic1.zhimg.com/v2-7b15707ebe7e520155d798e53f6fac34_r.jpg)
## L-softmax
L-softmax加强分类条件，强制让对应类别的W和x夹角增加到原来的m倍。
![image](https://pic4.zhimg.com/v2-6812d26f2c68955361960282f22f96a7_r.jpg)
![image](https://pic1.zhimg.com/v2-9175b2b838ef9082ba40983a4fc368f8_r.jpg)
## Sphere loss
在L-softmax的基础上，归一化了权值W，让训练更加集中在优化深度特征映射和特征向量角度上，降低样本数量不均衡问题。但是采用乘性margin较难收敛，m越大，越难收敛。
![image](https://pic4.zhimg.com/v2-5932cc2e4558e08d690e8ba2ce1bbb3b_r.jpg)
![image](https://pic1.zhimg.com/v2-9175b2b838ef9082ba40983a4fc368f8_r.jpg)
![image](https://github.com/LrpljL/FR-Loss/blob/master/Sphere/epoch%3D79.jpg)
![image](https://github.com/LrpljL/FR-Loss/blob/master/Sphere/cos_epoch%3D79.jpg)
## center loss
为每个类别学习一个中心，并将每个类别的所有特征向量拉向对应类别中心，联合softmax一起使用。
![image](https://pic4.zhimg.com/v2-eaaf34827c44c4c45085647962144a2f_r.jpg)
![image](https://github.com/LrpljL/FR-Loss/blob/master/Center/epoch%3D79.jpg)
![image](https://github.com/LrpljL/FR-Loss/blob/master/Center/cos_epoch%3D79.jpg)
## insight face
![image](https://pic3.zhimg.com/v2-24382d5345b7e442602f3be895e454fa_r.jpg)
![image](https://github.com/LrpljL/FR-Loss/blob/master/ArcSoft/epoch%3D79.jpg)
![image](https://github.com/LrpljL/FR-Loss/blob/master/ArcSoft/cos_epoch%3D79.jpg)
