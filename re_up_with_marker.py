from tello_sdk_stand import *
#   启用挑战卡
dj = Start()

try:
    # dj.led_obj.set_mled_sc()  # 关闭LED灯，节约用电
    dj.take_off()  # 起飞
    mon_status = True
    res = dj.command('mon')  # 打开识别定位卡功能
    if 'error' in res:
        print('打开识别定位卡功能')
        mon_status = False
    # res2 = dj.command('mdirection 0')  # 设置定位卡为下视识别  模式是下视
    # print('打开下视定位卡结果', res2)
    time.sleep(1)
    # 螺旋上升
    dj.curve(x1=85, y1=35, z1=20, x2=170, y2=0, z2=30, speed=30)
    dj.curve(x1=-85, y1=-40, z1=20, x2=-170, y2=20, z2=30, speed=30)
    # 环绕山峰
    dj.forward(200)  # 飞到山峰前
    dj.right(30) # 调整左右位置
    # dj.take_photo('before',num=4)
    dj.down(70) # 下降高度
    # 定位到距离悬崖60-80厘米的位置  ，智慧调整前后位置，不会调整左右
    cliff_loc(dj)
    # 悬停3秒
    time.sleep(3)

    # dj.take_photo('ing',num=4)
    dj.reverse(360)  # 旋转360°
    # dj.take_photo('after',num=4)
    # 飞跃悬崖
    dj.up(70)  # 上升
    dj.forward(290) # 向前飞
    # 特级表演
    dj.down(90)  # 下降高度
    dj.back(120) # 后退
    dj.up(90) # 上升
    dj.forward(130)  # 向前
    dj.right(100)  # 右飞
    dj.land() # 降落
except Exception as e:
    print('飞行异常准备降落，错误代码：', e)
    dj.land()  # 降落

dj.close()  # 释放飞机资源



