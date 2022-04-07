# 导入sdk
from tello_sdk_stand import *

# 拿到飞机控制权
dj = Start()

# 起飞
dj.take_off()
# 前 50  右50 后 100 降落     单位是厘米
# 向前飞50
dj.forward(50)
# 向右飞50
dj.right(50)
# 向后飞100
dj.back(100)
# 降落
dj.land()

# 释放资源
dj.close()
