# -*-coding:utf-8-*-
# Copyright (c) 2020 DJI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 大疆Tello TT 官方SDK
# 使用说明 详见官网 https://www.dji.com/cn/robomaster-tt
# 源代码已托管至github https://github.com/dji-sdk/RoboMaster-SDK/

import random
import sys
import threading
import cv2
from robomaster import robot, protocol, logger
from robomaster import camera
import time

# Press the green button in the gutter to run the script.
from yolov5_new.tt_api import get_qi_info, get_gan_info


class Start:
    def __init__(self):
        self.t1_drone = robot.Drone()
        self.t1_drone.initialize()
        self.led_obj = self.t1_drone.led
        self.flight_obj = self.t1_drone.flight
        # self.t1_led.set_led(r=0, g=255, b=0)
        self.camera_obj = self.t1_drone.camera
        self.sensor_obj = self.t1_drone.sensor
        self.battery_obj = self.t1_drone.battery
        print('wifi信噪比：', self.t1_drone.get_wifi())
        current_battery = self.battery_obj.get_battery()
        print('电池电量：', current_battery, '%')
        if current_battery <= 10:
            print('电量小于10%无法飞行')
            raise NameError('电量小于10%无法飞行')
        self.camera_obj.start_video_stream(display=False)

    def command(self, cmd=None):
        """
               执行自定义指令
               """

        if cmd:
            proto = protocol.TextProtoDrone()
            proto.text_cmd = cmd
            msg = protocol.TextMsg(proto)
            try:
                resp_msg = self.flight_obj._client.send_sync_msg(msg)
                if resp_msg:
                    proto = resp_msg.get_proto()
                    print('执行结果', proto)
                    return proto.resp
                else:
                    logger.warning("Drone: 发送指令失败 failed.")
                    return "Drone: 发送指令失败 failed."

            except Exception as e:
                logger.warning("Drone: flight.command, send_sync_msg exception {0}".format(str(e)))
                return "Drone: flight.command, send_sync_msg exception {0}".format(str(e))
        else:
            print('请输入指令')
            return '请输入指令'

    def change_led(self, r=0, g=0, b=0):
        self.led_obj.set_led(r, g, b)

    def take_off(self):
        """起飞"""
        self.flight_obj.takeoff().wait_for_completed()
        time.sleep(0.1)

    def land(self):
        """降落"""
        print('收到降落指令！')
        self.flight_obj.land().wait_for_completed(15)

    def up(self, distance=0, retry=True):
        """ 向上飞distance厘米，指相对距离

                :param: distance: float:[20, 500]向上飞行的相对距离，单位 cm
                :param: retry: bool:是否重发命令
                :return: action对象
                """
        self.flight_obj.up(distance=distance, retry=retry).wait_for_completed(15)

    def down(self, distance=0, retry=True):
        """ 向下飞distance厘米，指相对距离

        :param: distance: float:[20, 500]向下飞行的相对距离，单位 cm
        :param: retry: bool:是否重发命令
        :return: action对象
        """
        self.flight_obj.down(distance=distance, retry=retry).wait_for_completed(15)

    def forward(self, distance=0, retry=True, force=False):
        """ 向前飞行distance厘米，指相对距离

                :param: distance: float:[20, 500]向前飞行的相对距离，单位 cm
                :param: retry: bool:是否重发命令
                :param:force:True强制飞行，False 安全飞行
                :return: action对象
                """
        cur_dist = self.get_dist()
        if cur_dist < 15 or cur_dist < distance:
            self.land()
            raise NameError('前方安全距离不足无法飞行，已自动降落！！！')
            sys.exit()

        # time.sleep(1)
        self.flight_obj.forward(distance=distance, retry=retry).wait_for_completed(15)

    def back(self, distance=0, retry=True):
        """ 向后飞行distance厘米， 指相对距离

        :param: distance: float:[20, 500]向后飞行的相对距离，单位 cm
        :param: retry: bool:是否重发命令
        :return: action对象
        """
        self.flight_obj.backward(distance=distance, retry=retry).wait_for_completed(15)

    def left(self, distance=0, retry=True):
        """ 向左飞行distance厘米， 指相对距离

        :param: distance: float:[20, 500]向左飞行的相对距离，单位 cm
        :param: retry: bool:是否重发命令
        :return: action对象
        """
        self.flight_obj.left(distance=distance, retry=retry).wait_for_completed(15)

    def right(self, distance=0, retry=True):
        """ 向右飞行distance厘米， 指相对距离

        :param: distance: float:[20, 500]向右飞行的相对距离，单位 cm
        :param: retry: bool:是否重发命令
        :return: action对象
        """
        self.flight_obj.right(distance=distance, retry=retry).wait_for_completed(15)

    def fly(self, direction='forward', distance=0, retry=True):
        """ 控制飞机向指定方向飞行指定距离。

        :param: direction: string: 飞行的方向，"forward" 向前飞行， "back" 向后飞行， "up" 向上飞行，
                                    "down" 向下飞行， "left" 向左飞行， "right" 向右飞行
        :param: distance: float:[20, 500]，飞行的距离，单位 cm
        :param: retry: bool:是否重发命令
        :return: action对象
        """
        self.flight_obj.fly(direction=direction, distance=distance, retry=retry).wait_for_completed(15)

    def go(self, x, y, z, speed=10, mid=None, retry=True):
        """ 控制飞机以设置速度飞向指定坐标位置

               注意， x,y,z 同时在-20~20时，飞机不会运动。当不使用挑战卡时，飞机所在位置为坐标系原点，飞机的前方为x轴正方向，飞机的左方为y轴的正方向

               :param: x: float: [-500, 500] x轴的坐标，单位 cm
               :param: y: float: [-500, 500] y轴的坐标，单位 cm
               :param: z: float: [-500, 500] z轴的坐标，单位 cm
               :param: speed: float: [10, 100] 运动速度， 单位 cm/s
               :param: mid: string: 不使用挑战卡时mid为None，运动坐标系为飞机自身坐标系；当使用挑战卡时mid为对应挑战卡编号，
                                   运动坐标系为指定挑战卡的坐标系。支持编号可参考挑战卡使用说明。
               :param: retry: bool:是否重发命令
               :return: action对象
               """
        res = self.flight_obj.go(x=x, y=y, z=z, speed=speed, mid=mid, retry=retry).wait_for_completed(15)
        print('tello_sdk_stand.go()::res:::::res::::', res)
        return res

    def go_(self, x, y, z, speed=10, mid=None):
        """本函数为自定义，非djsdk提供   发送自定义命令会返回，执行结果"""
        print('调用自定义go函数,需要加等待')
        cmd = "go {0} {1} {2} {3}".format(x, y, z, speed)

        if mid:
            cmd += " {0}".format(mid)
        # return  self.flight_obj.command(cmd)
        return self.command(cmd)

    def curve(self, x1=0, y1=0, z1=0, x2=0, y2=0, z2=0, speed=20, mid=None, retry=True):
        """
        本函数为自定义，非dj sdk提供
        以设置速度飞弧线，经过对应坐标系中的(x1, y1, z1)点到（x2, y2, z2）点

        如果选用mid参数，则对应坐标系为指定挑战卡的坐标系。不使用挑战卡时，飞机的前方为x轴正方向，飞机的左方为y轴的正方向
        如果mid参数为默认值None,则为飞机自身坐标系
        """
        return self.flight_obj.curve(x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2, speed=speed, mid=mid, retry=retry).wait_for_completed(15)

    def curve_(self, x1=0, y1=0, z1=0, x2=0, y2=0, z2=0, speed=20, mid=None):
        """
        print('调用自定义curve_函数,需要加等待')
        本函数为自定义，非dj sdk提供
        以设置速度飞弧线，经过对应坐标系中的(x1, y1, z1)点到（x2, y2, z2）点

        如果选用mid参数，则对应坐标系为指定挑战卡的坐标系。不使用挑战卡时，飞机的前方为x轴正方向，飞机的左方为y轴的正方向
        如果mid参数为默认值None,则为飞机自身坐标系
        """
        cmd = ""
        if mid:
            cmd = "curve {0} {1} {2} {3} {4} {5} {6} {7}".format(
                x1, y1, z1, x2, y2, z2, speed, mid)
        else:
            cmd = "curve {0} {1} {2} {3} {4} {5} {6}".format(
                x1, y1, z1, x2, y2, z2, speed)
        # return  self.flight_obj.command(cmd)
        return self.command(cmd)

    def reverse(self, angle=0, retry=True):
        """ 控制飞机旋转指定角度

        :param: angle: float:[-360, 360] 旋转的角度，俯视飞机时，顺时针为正角度，逆时针为负角度
        :param: retry: bool:是否重发命令
        :return: action对象
        """
        self.flight_obj.rotate(angle=angle, retry=retry).wait_for_completed(15)

    def get_video(self):
        self.camera_obj.start_video_stream(display=True)
        time.sleep(5)
        self.camera_obj.stop_video_stream()

    # def take_photo2(self, pre='dj', name='', num=2):
    #     """拍照此函数由问题，"""
    #     # self.camera_obj.set_fps('low')
    #     # self.camera_obj.set_resolution('high')
    #     print('调用拍照函数，即将开始循环拍照')
    #     for x in range(5):
    #         print(f'第{x}次外层循环开始')
    #         if name == '':
    #             pre = f"{pre}-{int(time.time())}-{random.randint(10, 100)}-"
    #         self.camera_obj.start_video_stream(display=False)
    #         print('内部开始拍照')
    #         # time.sleep(1)
    #         print('开始取流')
    #         img = self.camera_obj.read_cv2_image()
    #         for i in range(num):
    #             cv2.imwrite(f'./img/{pre}{i}.jpg', img)
    #             time.sleep(1)
    #             print(f'拍第{i}张照片完成')
    #         print('内部拍照结束！！！')
    #         # 停止取流有问题
    #         self.camera_obj.stop_video_stream()
    #         print('停止取流')
    #     print('外层循环结束')

    def take_photo(self, pre='dj', name='', num=2, wtime=0.5):
        """拍照函数，你要初始化中，初始化取流信息，self.camera_obj.start_video_stream(display=False)"""
        print('拍照开始')
        if name == '':
            name = f"{pre}-{int(time.time())}-{random.randint(10, 100)}-"
        for i in range(num):
            time.sleep(wtime)
            print('准备拍照')
            img = self.camera_obj.read_cv2_image(strategy='newest')
            img_path = f'./img/{name}{i}.jpg'
            # print(img_path)
            cv2.imwrite(img_path, img)
            print(f"第{i}次拍照完毕！！")
        return img_path

    def get_dist(self, num=3, wtime=0.2):
        """获取距离,返回值大于780可能不准确
        :param  num 测距次数
        :param  间隔时间
        :return 返还平均距离
        """
        # print('测距')
        sum_dis = 0.0
        l_dis = []
        for x in range(num):
            l_dis.append(self.sensor_obj.get_ext_tof())
            time.sleep(wtime)
        l_dis.sort()
        l_dis.pop(0)
        l_dis.pop(-1)
        for i in l_dis:
            sum_dis += i
        avg_dis = int((sum_dis / len(l_dis)) / 10)
        print('平均距离:', avg_dis)
        return avg_dis

    def close(self):
        """释放无人机资源"""
        print('释放资源')
        self.t1_drone.close()


def qi_loc(task, step=0, x=0, step_y=0, move_y_status=False):
    """定位旗杆的位置
     :param move_y_status: y轴方向移动状态，移动后被设置为True
     :param task:  飞机对象
    :param step 移动的步数，美动一次加1
    :param  x 当前再x坐标轴的位置
    :param step_y 向y轴移动的次数初始值
    ;return  返回  定位是否成功和 障碍物距离  [True, cur_dist]"""

    print(f'当前移动次数{step},当前位置:{x},{move_y_status}')
    cur_dist = task.get_dist()
    d_stand = 780  # 测距零界点
    move_x = 30  # 向x轴移动的距离
    max_x = 150  #
    key_step = max_x / move_x
    move_y = 100  # 向y轴移动的距离
    step_y_stand = 1  # 最多向y轴方向移动的次数,最大支持2次
    if cur_dist < d_stand:
        print(f'定位成功，距离：{cur_dist},当前位置{x},当前移动次数{step}')
        return [True, cur_dist]
    else:
        if x > 0:
            # 已回到原点
            print('定位失败，回到原点：')
            return [False, 900]
        elif x <= 0 and x > -max_x and step < key_step:
            print('准备向左移动')
            # 小于80 向右移右20
            task.left(move_x)
            step += 1
            x = x - move_x
            cur_dist = task.get_dist()
            if cur_dist < d_stand:
                print(f'定位成功，距离：{cur_dist},位置{x}，次数{step}')
                return [True, cur_dist]
            else:
                # 继续向右移动
                move_y_status = False
                return qi_loc(task, step, x, step_y, move_y_status)
        elif x < 0 and step >= key_step and step < key_step * 2:
            if x == -max_x and step_y < step_y_stand and not move_y_status:
                print(f'准备向前移动{move_y}，第{step_y}次向前移动')
                task.forward(move_y)
                move_y_status = True
                step_y = step_y + 1
                cur_dist = task.get_dist()
                if cur_dist < d_stand:
                    print(f'定位成功，距离：{cur_dist},位置{x}，次数{step}')
                    return [True, cur_dist]
                else:
                    # 需要继续向右移动
                    return qi_loc(task, step, x, step_y, move_y_status)
            else:
                print('准备向右移动')
                task.right(move_x)
                step += 1
                x += move_x
                cur_dist = task.get_dist()
                if cur_dist < d_stand:
                    print(f'定位成功，距离：{cur_dist},位置{x}，次数{step}')
                    return [True, cur_dist]
                else:
                    # 继续向右移动
                    move_y_status = False
                    return qi_loc(task, step, x, step_y, move_y_status)
        elif x <= 0 and x > -max_x and step >= key_step * 2 and step < key_step * 3:
            if x == 0 and step_y < step_y_stand and not move_y_status:
                print(f'准备向前移动{move_y}，第{step_y}次向前移动')
                task.forward(move_y)
                move_y_status = True
                step_y = step_y + 1
                cur_dist = task.get_dist()
                if cur_dist < d_stand:
                    print(f'定位成功，距离：{cur_dist},位置{x}，次数{step}')
                    return [True, cur_dist]
                else:
                    # 需要继续向右移动
                    return qi_loc(task, step, x, step_y, move_y_status)
            else:
                print('准备向左移动')
                # 小于80 向右移右20
                task.left(move_x)
                step += 1
                x = x - move_x
                cur_dist = task.get_dist()
                if cur_dist < d_stand:
                    print(f'定位成功，距离：{cur_dist},位置{x}，次数{step}')
                    return [True, cur_dist]
                else:
                    # 继续向右移动
                    move_y_status = False
                    return qi_loc(task, step, x, step_y, move_y_status)
        else:
            print(f'定位失败，位置异常，当前位置：{x},当前移动次数{step}')
            return [False, 900]


def get_gan_loc(model, dj, x_step=-40, y_step=-20, try_num=10, take_photo_num=2):
    """杆定位
    model: 杆识别模型
    dj:飞机对象
    x_step:x每次向x方向移动的距离，左手法则，负数向左移动
    y_step：每次前后移动的距离， 左手法则
    try_num:寻找杆的次数
    take_photo_num:拍张张数
    返回：未找到返回not found ,找到后返杆相对飞机的，前方距离，和飞机左右方向距离
    """
    try_num = try_num - 1
    result = dict()
    result['code'] = 'not found'
    img_path = dj.take_photo(num=take_photo_num)
    take_photo_num = 1
    print('图片路径：', img_path)
    gan_info = get_gan_info(model, img_path)
    print(gan_info)
    if gan_info['code'] == 'action':
        dj.fly(direction=gan_info['direction'], distance=gan_info['distance'])
        if not gan_info['finish']:
            # img_path = dj.take_photo(num=take_photo_num)
            # gan_info = get_gan_info(model, img_path)
            # dj.fly(direction=gan_info['direction'], distance=gan_info['distance'])
            print('需要再次调用拍照定位')
            get_gan_loc(model, dj, x_step, y_step, try_num=try_num, take_photo_num=1)
    elif gan_info['code'] == 'no action':
        result['code'] = 'found'
        result['msg'] = '找到杆位置'
        result['distance'] = gan_info['distance']  # 左手法则，负数向左飞，正数向右飞
        result['dis_forward'] = gan_info['dis_forward']
        print('无需调整位置')
    else:
        # code 为err
        if not gan_info['finish'] and try_num > 0:
            print('未检测到杆，想右 向后飞行后再试')
            if abs(x_step) >= 20:
                # 左手法则，负数向左飞
                if try_num % 4 == 0:
                    # 每4次调整一次方向
                    x_step = -x_step
                x_direction = 'left' if x_step < 0 else 'right'
                dj.fly(direction=x_direction, distance=int(x_step))
            if abs(y_step) >= 20 and try_num % 4 == 0:
                # 左手法则，正数数向前飞，每四次调整一次前后位置，默认向后
                y_direction = 'back' if y_step < 0 else 'forward'
                dj.fly(direction=y_direction, distance=int(y_step))
            get_gan_loc(model, dj, x_step, y_step, try_num=try_num, take_photo_num=1)
        else:
            logger.warning('尝试次数已用完，未找到杆')
            logger.warning(gan_info)
    return result


def get_qi_loc(model, dj, x_step=-40, y_step=-40, try_num=10, take_photo_num=2):
    """旗子定位
    model: 旗子识别模型
    dj:飞机对象
    x_step:x每次向x方向移动的距离，左手法则，负数向左移动
    y_step：每次前后移动的距离， 左手法则
    try_num:寻找旗子的次数
    take_photo_num:拍张张数
    返回：未找到返回not found ,找到后返旗子相对飞机的，前方距离，和飞机左右方向距离
    """
    try_num = try_num - 1
    result = dict()
    result['code'] = 'not found'
    img_path = dj.take_photo(num=take_photo_num)
    take_photo_num = 1
    print('图片路径：', img_path)
    qi_info = get_qi_info(model, img_path)
    print(qi_info)
    if qi_info['code'] == 'action':
        dj.fly(direction=qi_info['direction'], distance=qi_info['distance'])
        if not qi_info['finish']:
            # img_path = dj.take_photo(num=take_photo_num)
            # qi_info = get_qi_info(model, img_path)
            # dj.fly(direction=qi_info['direction'], distance=qi_info['distance'])
            print('需要再次调用拍照定位')
            get_qi_loc(model, dj, x_step, y_step, try_num=try_num, take_photo_num=1)
    elif qi_info['code'] == 'no action':
        result['code'] = 'found'
        result['msg'] = '找到旗子位置'
        result['distance'] = qi_info['distance']  # 左手法则，负数向左飞，正数向右飞
        result['dis_forward'] = qi_info['dis_forward']
        print('无需调整位置')
    else:
        # code 为err
        if not qi_info['finish'] and try_num > 0:
            print('未检测到旗子，想右 向后飞行后再试')
            if abs(x_step) >= 20:
                # 左手法则，负数向左飞
                if try_num % 4 == 0:
                    # 每4次调整一次方向
                    x_step = -x_step
                x_direction = 'left' if x_step < 0 else 'right'
                dj.fly(direction=x_direction, distance=int(x_step))
            if abs(y_step) >= 20 and try_num % 4 == 0:
                # 左手法则，正数数向前飞，每四次调整一次前后位置，默认向后
                y_direction = 'back' if y_step < 0 else 'forward'
                dj.fly(direction=y_direction, distance=int(y_step))
            get_qi_loc(model, dj, x_step, y_step, try_num=try_num, take_photo_num=1)
        else:
            logger.warning('尝试次数已用完，未找到旗子')
            logger.warning(qi_info)
    return result


def gan_loc(task, step, x):
    """定位杆的位置
    :param task:  飞机对象
    :param step 移动的步数，美动一次加1
    :param  x 当前再x坐标轴的位置
    """
    print(f'当前移动次数{step},{type(step)},当前位置:{x}')

    cur_dist = task.get_dist()
    d_stand = 780
    if cur_dist < d_stand:
        print(f'定位成功，距离：{cur_dist},当前位置{x},当前移动次数{step}')
        return [True, cur_dist]
    else:
        if (x + 20) == 0:
            # 已回到原点
            print('定位失败，回到原点：')
            return [False, 900]
        elif (step < 5 and x >= 0 and x < 110) or (step >= 15 and x < -10):
            # elif (step < 5 and 0 <= x < 110) or (step > 15 and x < 0):
            # 移动次数小于第五步，并且x轴的位置在0到110之间向有移动
            print('准备向右移动')
            move_x = 20 if x < 80 else 30
            # 小于80 向右移右20
            task.right(move_x)
            step += 1
            x += move_x
            cur_dist = task.get_dist()
            if cur_dist < d_stand:
                print(f'定位成功，距离：{cur_dist},位置{x}，次数{step}')
                return [True, cur_dist]
            else:
                # 继续向右移动
                return gan_loc(task, step, x)
        elif step >= 5 and step < 15 and x > -100 and x <= 110:
            # elif step >=5 and step <=15:
            # elif (5 >= step < 15) and (-100 > x <= 110):
            # 向左移动
            print('准备向左移动')
            move_x = 20 if x >= -70 else 30
            # 小于80 向右移右20
            task.left(move_x)
            step += 1
            x -= move_x
            cur_dist = task.get_dist()
            if cur_dist < d_stand:
                print(f'定位成功，距离：{cur_dist},位置{x}，次数{step}')
                return [True, cur_dist]
            else:
                # 继续向右移动
                return gan_loc(task, step, x)
        else:
            print(f'定位失败，位置异常，当前位置：{x},当前移动次数{step}')
            return [False, 900]


#

def qi_fei(dj):
    """起飞"""
    dj.led_obj.set_mled_sc()  # 关闭LED灯，节约用电
    dj.take_off()  # 起飞
    dj.down(20)  # 下降高度


def guo_men(dj):
    """过埃门，   下一步绕杆 rao_gan(dj)"""

    dj.forward(240)  # 过矮门

    dj.up(70)  # 上升高度 准备过高门
    dj.forward(200)  # 过高门


def rao_gan(dj):
    """绕杆，并向右移动，   下一步改定位旗杆位置并绕旗飞行"""
    dj.right(70)  # 开始绕杆
    dj.forward(140)  # 向前飞
    dj.left(140)  # 向左飞
    dj.back(140)  # 向后飞
    # 绕杆结束，调整位置,准备绕第旗杆
    dj.right(220)


def rao_qi(dj):
    """绕旗杆"""

    #  检测前方是否由旗杆如果有先向左飞50厘米，然后根据情况计算向前飞距离
    juli = dj.get_dist()  # 监测前方是否有障碍物
    qian_jin_ju_li = 170  # 设置标桩前进距离
    if juli < 780:  # 若果前方有障碍物
        dj.left(50)  # 向左移动，避开障碍物
        if juli < 200:
            qian_jin_ju_li = juli + 60  # 前进距离等于 距离障碍物的距离+60
    dj.forward(qian_jin_ju_li)  # 向前进

    dj.xuanzhuan(180)  # 调头
    # 向左飞行  ！！可能需要调整
    dj.left(50)  # 向左飞行，  如果出现飞多了监测不到旗杆，需要减少
    dingwei_jieguo = qi_loc(dj)  # 寻找旗杆的位置，先向左找1.5米，前进90厘米，再向右边找
    if dingwei_jieguo[0]:
        # 定位成功
        # dj.down(70)  #
        dj.left(90)
        dj.forward(int(dingwei_jieguo[1] + 80))  # 前进距离 = 距离旗杆距离 + 80
    else:
        # 定位失败，如果需要继续盲飞，请修改此处
        raise NameError('旗杆定位失败，请求降落')
    dj.right(140)  # 向右飞
    dj.forward(140)  # 向前飞
    dj.left(130)  # 向左飞
    dj.forward(220)  # 向前飞


def pai_zhao(dj):
    """拍照"""
    dj.xuanzhuan(90)
    dj.down(30)
    dj.forward(200)
    dj.xuanzhuan(90)
    dj.take_photo()
    dj.forward(100)

    # 循环左右拍照
    for i in range(4):  # 如需修改 循环次数，修改括号里面的值
        dj.take_photo()
        dj.xuanzhuan(-45)
        dj.take_photo()
        dj.xuanzhuan(90)
        dj.take_photo()
        dj.xuanzhuan(-45)
        dj.forward(100)  # 拍照距离间隔
        dj.take_photo()

    # 掉头继续循环拍照
    dj.xuanzhuan(180)  # 掉头

    for i in range(4):  # 如需修改 循环次数，修改括号里面的值
        dj.take_photo()
        dj.xuanzhuan(-45)
        dj.take_photo()
        dj.xuanzhuan(90)
        dj.take_photo()
        dj.xuanzhuan(-45)
        dj.forward(90)  # 拍照距离间隔
        dj.take_photo()


def jiang_luo(dj):
    """jiang_luo()  # 降落"""
    dj.land()
