# FR-Loss
人脸识别损失函数对比总结
参考知乎链接：
https://zhuanlan.zhihu.com/p/34404607<br>https://zhuanlan.zhihu.com/p/34436551
## Softmax 
softmax只保证能够正确的分类，但是无法保证同一类特征之间类内紧致，类间间隔大，如下图：
![image](https://github.com/LrpljL/FR-Loss/blob/master/Raw/cos_epoch%3D79.jpg)
![image](https://github.com/LrpljL/FR-Loss/blob/master/Raw/epoch%3D79.jpg)
## Sphere loss
在L-softmax的基础上，归一化了权值W，让训练更加集中在优化深度特征映射和特征向量角度上，降低样本数量不均衡问题。采用乘性margin
![image](https://pic2.zhimg.com/v2-7ae3dafcbc9c0aff160f6718e1e0b769_r.jpg)
