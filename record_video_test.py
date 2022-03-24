import os
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
# try:
video_writer = cv2.VideoWriter('img/video/xx.avi', cv2.VideoWriter_fourcc(*'XVID'), 15.0, (960, 720))
for i in range(0,100):
    video_writer.write(dj.camera_obj.read_video_frame())
    time.sleep(0.1)
    print(i)
video_writer.release()
# os.system(f'ffmpeg -i "img/video/xx.avi" -vcodec h264 "img/video/xx.mp4"')
dj.camera_obj.stop_video_stream()
# except Exception as e:
#     print(e)
dj.close()