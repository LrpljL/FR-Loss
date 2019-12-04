# FR-Loss
人脸识别损失函数对比总结
参考知乎链接：
https://zhuanlan.zhihu.com/p/34404607<br>https://zhuanlan.zhihu.com/p/34436551
## softmax 
softmax只保证能够正确的分类，但是无法保证同一类特征之间类内紧致，类间间隔大，如下图：
![image](https://github.com/LrpljL/FR-Loss/blob/master/Raw/cos_epoch%3D79.jpg)
