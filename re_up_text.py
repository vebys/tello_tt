import time
import datetime
import random
from  tello_sdk_stand import *
from robomaster import led
import sys
import threading
import cv2

"""先过矮门，再过高门，从左边绕旗杆"""



dj = Start()



try:
    """起飞"""
    dj.led_obj.set_mled_sc()  # 关闭LED灯，节约用电
    dj.take_off()  # 起飞
    #
    dj.down(20)
    dj.flight_obj.curve(x1=-30, y1=70, z1=40, x2=70, y2=170, z2=80 , speed=20)
except Exception as e:
    print('飞行异常准备降落，错误代码：', e)
    dj.land()  # 降落
finally:
    dj.close()  # 释放飞机资源
    sys.exit()