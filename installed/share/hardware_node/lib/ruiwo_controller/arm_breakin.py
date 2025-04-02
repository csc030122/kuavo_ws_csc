import os
import yaml
import time
from SimpleSDK import RUIWOTools

# 速度相关参数，方便调整
MOTION_DURATION = 0.35  # 每个动作的执行时间（秒）
VEL = 0
POS_KP = 20
POS_KD = 3
TORQUE = 0

# 500Hz 的更新频率
UPDATE_FREQUENCY = 500  # Hz
UPDATE_INTERVAL = 1 / UPDATE_FREQUENCY  # 秒

# 定义零点文件路径
def get_zero_path():
    return '/home/lab/.config/lejuconfig/arms_zero.yaml'

# 读取零点位置
def read_zero_positions():
    zeros_path = get_zero_path()
    if os.path.exists(zeros_path):
        with open(zeros_path, 'r') as file:
            zeros_config = yaml.safe_load(file)
        return zeros_config['arms_zero_position']
    else:
        print("[RUIWO motor]:Warning: zero_position file does not exist, will use 0 as zero value.")
        return [0.0] * 14

# 获取用户输入的测试时长
def get_test_duration():
    while True:
        try:
            duration = int(input("请输入测试时长（5 - 600 秒）："))
            if 5 <= duration <= 600:
                return duration
            else:
                print("输入的时长不在 5 - 600 秒范围内，请重新输入。")
        except ValueError:
            print("输入无效，请输入一个整数。")

# 获取当前时间戳
def get_timestamp():
    return time.strftime("%H:%M:%S", time.localtime()) + f".{int(time.time() % 1 * 1000):03d}"

# 读取电机正反转配置
def read_motor_reverse_config():
    config_path = '/home/lab/.config/lejuconfig/config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config.get('negtive_address', [])
    else:
        print("[RUIWO motor]:Warning: config.yaml file does not exist, no motor reverse config will be applied.")
        return []

# 创建对象
ruiwo = RUIWOTools()

# 打开CAN总线
open_canbus = ruiwo.open_canbus()
if not open_canbus:
    print("[RUIWO motor]:Canbus status:","[",open_canbus,"]")
    exit(1)
print("[RUIWO motor]:Canbus status:","[",open_canbus,"]")

# 关节 ID 列表
joint_ids = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C]

# 读取零点位置
zero_positions = read_zero_positions()

# 读取电机正反转配置
reverse_addresses = read_motor_reverse_config()

# 初始化关节正反转标志列表
joint_directions = [1] * len(joint_ids)
for index, dev_id in enumerate(joint_ids):
    if dev_id in reverse_addresses:
        joint_directions[index] = -1

# 使能所有关节电机
enable_all_success = True  # 用于标记是否所有电机都使能成功
for dev_id in joint_ids:
    state = ruiwo.enter_motor_state(dev_id)
    if isinstance(state, list):
        print(f"[RUIWO motor]:ID: {dev_id} Enable:  [Succeed]")
    else:
        print(f"[RUIWO motor]:ID: {dev_id} Enable:  [{state}]")
        enable_all_success = False  # 只要有一个电机使能失败，就标记为失败
if not enable_all_success:
    print("有电机使能失败，程序退出。")
    exit(1)  # 退出程序

base_actions = [
    [0.23, -0.00, 0.00, -0.00, 0.00, -0.24,     -0.23, -0.00, -0.00, -0.00, -0.00, 0.24],
    [1.30, 1.00, -1.40, -1.33, 0.40, -0.77,     -1.30, -1.00, 1.40, 1.33, -0.40, 0.77],
    [1.94, -0.50, -0.37, 1.51, 0.76, 0.76,      -1.94, 0.50, 0.37, -1.51, -0.76, -0.76],
    [1.09, -1.70, -0.93, -1.51, 0.00, -0.24,    -1.09, 1.70, 0.93, 1.51, -0.00, 0.24],
    [0.23, -0.40, -1.93, 1.07, -0.70, 0.73,     -0.23, 0.40, 1.93, -1.07, 0.70, -0.73],
    [2.05, -0.00, -0.75, 0.71, 0.00, -0.52,     -2.05, -0.00, 0.75, -0.71, -0.00, 0.52],
    [0.23, -0.00, 0.00, -0.00, 0.00, -0.24,     -0.23, -0.00, -0.00, -0.00, -0.00, 0.24],
]

# 执行动作序列
action_index = 0

# 获取用户输入的测试时长
test_duration = get_test_duration()
start_time = time.perf_counter()

while True:
    current_time = time.perf_counter()
    elapsed_time = current_time - start_time
    remaining_time = test_duration - elapsed_time

    if remaining_time <= 3:
        # 剩余时间不足 3 秒，回到位置 1
        target_positions = base_actions[0]  # 位置 1 的动作
        current_positions = base_actions[action_index - 1] if action_index > 0 else [0.0] * len(joint_ids)
        steps = MOTION_DURATION * UPDATE_FREQUENCY  # 按照 MOTION_DURATION 计算步数

        for step in range(int(steps)):
            loop_start = time.perf_counter()
            for joint_index, dev_id in enumerate(joint_ids):
                # 计算当前位置到目标位置的插值
                interpolated_pos = current_positions[joint_index] + (target_positions[joint_index] - current_positions[joint_index]) * (step / steps)
                zero_position = zero_positions[joint_index]
                compensated_pos = interpolated_pos * joint_directions[joint_index] + zero_position  # 应用正反转标志和零点补偿
                state = ruiwo.run_ptm_mode(dev_id, compensated_pos, VEL, POS_KP, POS_KD, TORQUE)
                if isinstance(state, list):
                    pass
                else:
                    print(f"{get_timestamp()} ID: {dev_id} Run ptm mode:  [{state}]")

            loop_end = time.perf_counter()
            elapsed_time_loop = loop_end - loop_start
            remaining_time_loop = UPDATE_INTERVAL - elapsed_time_loop
            if remaining_time_loop > 0:
                time.sleep(remaining_time_loop)
        break

    if elapsed_time >= test_duration:
        break

    print(f"{get_timestamp()} 现在是运行到第 {elapsed_time:.2f} 秒，还剩 {remaining_time:.2f} 秒，开始执行动作 {action_index + 1}")

    current_positions = base_actions[action_index - 1] if action_index > 0 else [0.0] * len(joint_ids)
    target_positions = base_actions[action_index]

    # 检查长度是否匹配
    if len(current_positions) != len(joint_ids) or len(target_positions) != len(joint_ids):
        raise ValueError(f"动作 {action_index + 1} 的位置列表长度不匹配。当前长度: {len(current_positions)}，目标长度: {len(target_positions)}，关节数量: {len(joint_ids)}")

    steps = MOTION_DURATION * UPDATE_FREQUENCY  # MOTION_DURATION 秒内发送的步数

    action_start_time = time.perf_counter()
    for step in range(int(steps)):
        loop_start = time.perf_counter()
        for joint_index, dev_id in enumerate(joint_ids):
            # 计算当前位置到目标位置的插值
            interpolated_pos = current_positions[joint_index] + (target_positions[joint_index] - current_positions[joint_index]) * (step / steps)
            zero_position = zero_positions[joint_index]
            compensated_pos = interpolated_pos * joint_directions[joint_index] + zero_position  # 应用正反转标志和零点补偿
            state = ruiwo.run_ptm_mode(dev_id, compensated_pos, VEL, POS_KP, POS_KD, TORQUE)
            if isinstance(state, list):
                pass
            else:
                print(f"{get_timestamp()} ID: {dev_id} Run ptm mode:  [{state}]")

        loop_end = time.perf_counter()
        elapsed_time = loop_end - loop_start
        remaining_time = UPDATE_INTERVAL - elapsed_time
        if remaining_time > 0:
            time.sleep(remaining_time)

    action_end_time = time.perf_counter()
    actual_duration = action_end_time - action_start_time
    print(f"{get_timestamp()} 动作 {action_index + 1} 实际执行时间: {actual_duration:.3f} 秒")

    # 输出当前动作完成后的状态
    next_action_index = action_index + 1
    if next_action_index < len(base_actions):
        print(f"{get_timestamp()} 现在到达位置 {action_index + 1}，开始向位置 {next_action_index + 1} 运动")
    else:
        action_index = 0  # 循环执行动作序列
        print(f"{get_timestamp()} 现在到达位置 {action_index + 1}，重新开始动作序列")

    action_index += 1

# 失能所有关节电机
for dev_id in joint_ids:
    state = ruiwo.enter_reset_state(dev_id)
    if isinstance(state, list):
        print(f"{get_timestamp()} [RUIWO motor]:ID: {dev_id} Disable:  [Succeed]")
    else:
        print(f"{get_timestamp()} [RUIWO motor]:ID: {dev_id} Disable:  [{state}]")

# 关闭CAN总线
close_canbus = ruiwo.close_canbus()
if close_canbus:
    print(f"{get_timestamp()} [RUIWO motor]:Canbus status: [Close]")