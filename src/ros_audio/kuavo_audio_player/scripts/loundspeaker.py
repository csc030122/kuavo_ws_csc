#!/usr/bin/env python3
import rospy
from kuavo_audio_player.srv import playmusic, playmusicResponse, playmusicRequest
import os
import subprocess  # 引入 subprocess 模块
import time

# 从ROS参数服务器获取音频路径参数，如果未设置则使用默认值
audio_path = rospy.get_param('audio_path', '/home/lab/.config/lejuconfig/music')
rospy.loginfo("音频文件路径: %s", audio_path)

class MusicPlayerNode:
    def __init__(self):
        while not self.check_sound_card():
            print("未检测到播音设备，不启用播音功能！")
            time.sleep(10000)
        print("启动music_player_node节点")
        rospy.init_node('music_player_node')
        self.service = rospy.Service('play_music', playmusic, self.play_music_callback)
        self.music_directory = audio_path  # 替换为你的音乐文件目录

    def check_sound_card(self):
        """
        检查声卡状态，特别是耳机和扬声器的可用性
        """
        # 检查耳机状态
        try:
            headphone_command = 'pactl list | grep -i Headphone'
            headphone_result = subprocess.run(headphone_command, shell=True, capture_output=True, text=True)
            print(headphone_result.stdout)
            if not bool(headphone_result.stdout.strip()):
                print(f"不存在耳机设备")
            # 检查耳机是否不可用
            else:
                headphone_available = "not available" not in headphone_result.stdout
                print(f"耳机状态: {'可用' if headphone_available else '不可用'}")
                if headphone_available:
                    return True
            # 检查扬声器状态
            speaker_command = 'pactl list | grep -i Speaker'
            speaker_result = subprocess.run(speaker_command, shell=True, capture_output=True, text=True)
            
            # 检查扬声器是否存在
            speaker_exists = bool(speaker_result.stdout.strip())
            print(f"扬声器状态: {'存在' if speaker_exists else '不存在'}")
            if speaker_exists:
                return True
            
            # root用户下检查扬声器状态
            root_speaker_command = 'aplay -l | grep -i Audio'
            root_speaker_result = subprocess.run(root_speaker_command, shell=True, capture_output=True, text=True)
            print(root_speaker_result.stdout)
            root_speaker_exists = bool(root_speaker_result.stdout.strip())
            print(f"root扬声器状态: {'存在' if root_speaker_exists else '不存在'}")
            if not root_speaker_exists:
                print(f"不存在扬声器设备")
            else:
                return True
            
            return False
            
        except Exception as e:
            print(f"检查声卡状态时出错: {str(e)}")
            return False

    def play_music_callback(self, req):
        try:
            # 获取服务请求的音乐文件序号和声音大小
            music_number = req.music_number
            volume = req.volume

            # 构建音乐文件路径
            music_file = os.path.join(self.music_directory, f"{music_number}")

            # 初始化设备及调整音量
            # setting_command = ['pactl', 'set-default-sink', '2']
            # subprocess.call(setting_command)
            # volume_command = ['pactl', 'set-sink-volume', '2', f'{volume}%']
            # subprocess.call(volume_command)

            # 使用 subprocess 调用 play 命令播放音乐
            play_command = ['play', '-q', music_file]
            subprocess.call(play_command)

            rospy.loginfo(f"Playing music {music_file} with volume {volume}%")
            
            # 创建 playmusic 类型的响应
            response = playmusicResponse()
            response.success_flag = True

            return response

        except Exception as e:
            rospy.logerr(f"Error playing music: {str(e)}")
            response = playmusicResponse()
            response.success_flag = False
            return response

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    player_node = MusicPlayerNode()
    player_node.run()