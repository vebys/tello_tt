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
    mon_status = True
    res = dj.command('mon')  # 打开识别定位卡功能
    if 'error' in res:
        print('打开识别定位卡功能')
        mon_status = False
    # res2 = dj.command('mdirection 0')  # 设置定位卡为下视识别


    dj.down(20)  # 下降高度



    """过埃门，   下一步绕杆 rao_gan(dj)"""

    dj.forward(230)  # 过矮门

    dj.up(70)  # 上升高度 准备过高门
    dj.forward(210)  # 过高门



    """绕杆，并向右移动，   下一步改定位旗杆位置并绕旗飞行"""
    dj.right(90)  # 开始绕杆
    dj.forward(140)  # 向前飞
    dj.left(140)  # 向左飞
    dj.back(150)  # 向后飞
    # 绕杆结束，调整位置,准备绕第旗杆
    dj.right(260)



    """绕旗杆"""

    #  检测前方是否由旗杆如果有先向左飞50厘米，然后根据情况计算向前飞距离
    juli = dj.get_dist()  # 监测前方是否有障碍物
    qian_jin_ju_li = 180  # 设置标桩前进距离
    if juli < 780:  # 若果前方有障碍物
        dj.left(50)  # 向左移动，避开障碍物
        if juli < 200:
            qian_jin_ju_li = juli + 60  # 前进距离等于 距离障碍物的距离+60
    dj.forward(qian_jin_ju_li)  # 向前进

    dj.reverse(180)  # 调头
    # 向左飞行  ！！可能需要调整
    dj.left(50)  # 向左飞行，  如果出现飞多了监测不到旗杆，需要减少
    dingwei_jieguo = qi_loc(dj)  # 寻找旗杆的位置，先向左找1.5米，前进90厘米，再向右边找
    if dingwei_jieguo[0]:
        # 定位成功
        # dj.down(70)  #
        dj.left(90)
        dj.forward(int(dingwei_jieguo[1] + 80))  # 前进距离 = 距离旗杆距离 + 80
    else:
        # 定位失败，如果需要继续盲飞，请修改此处,可以加定位卡
        raise NameError('找旗杆失败，请求降落')
    dj.right(170)  # 向右飞
    dj.forward(120)  # 向前飞
    dj.left(140)  # 向左飞    #原本 130
    dj.forward(120)  # 向前飞
    dj.right(60)  #向右飞行

    """寻找定位卡"""
    # 找定位卡
    if mon_status:
        for i in range(6):
            print('第', i, '次定位尝试')
            res_loc = dj.go(0, 0, 60, 10, 'm-2')
            print('定位结果：：' * 9, res_loc)
            if 'No valid marker' not in str(res_loc) and 'error' not in str(res_loc):
                print('定位成功！！！！！！！！！！！！！！！！！！！！！！！！！！！' * 2)
                # 定位成功
                dj.up(50)  # 上升40  理论上距离地面高度100cm
                dj.forward(140,retry=True,force=True)  # 穿圈

                """寻找终点定位卡"""
                for j in range(4):
                    print('第', j, '次找终点定位尝试')
                    res_loc_last = dj.go(0, 0, 100, 10, 'm-2')
                    if 'No valid marker' not in str(res_loc) and 'error' not in str(res_loc_last):
                        print('定位终点成功！准备降落！！' * 2)
                        break
                    dj.forward(30) # 未找到定位卡 向前飞30继续寻找
                else:
                    dj.back(30) # 找不到终点定位卡 向后退30准备降落
                break

            if i < 3:
                dj.right(50)
            else:
                dj.left(60)
    else:
        raise NameError('定位卡识别功能为开启，无法继续，请求降落')

    """验证是否定位成功"""

    dj.land()


except Exception as e:
    print('异常：', e)
    if '安全距离不足' not in str(e):
        print('代码报错：无法飞行，准备降落')
        dj.land()


finally:
    print('剩余电量：：',dj.battery_obj.get_battery())
    dj.close()
