本项目引用了作者qqwweee的kears-yolo3程序
作者链接：https://github.com/qqwweee/keras-yolo3
为了提升处理速度，在yolo3的基础上添加了opencv处理程序

检测部分
yolo3提供检测功能，代码没有改动

跟踪部分
详情请见yolo.py文件下的两段代码，大概为将检测到的目标位置送进kcf跟踪器中实现隔帧检测

项目运行
直接运行“运行程序.py”即可
