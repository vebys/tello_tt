import time
import datetime
import random
from  tello_sdk_stand import *
from robomaster import led
import sys
import threading
import cv2

"""�ȹ����ţ��ٹ����ţ�����������"""



dj = Start()



try:
    """���"""
    dj.led_obj.set_mled_sc()  # �ر�LED�ƣ���Լ�õ�
    dj.take_off()  # ���
    #
    dj.down(20)
    dj.flight_obj.curve(x1=-30, y1=70, z1=40, x2=70, y2=170, z2=80 , speed=20)
except Exception as e:
    print('�����쳣׼�����䣬������룺', e)
    dj.land()  # ����
finally:
    dj.close()  # �ͷŷɻ���Դ
    sys.exit()