import paramiko
import threading
import signal
import time
import sys

# 定义两个服务的配置（可扩展为类或字典）
pkill_command = "pkill -f ros"

SERVICES = [
    {
        "name": "play_music",
        "host": "192.168.26.1",
        "port": 22,
        "username": "kuavo",
        "password": "leju_kuavo",
        "remote_path": "/home/kuavo/kuavo_ros_application",
        "pkill_cmd": "pkill -f ros",  # 通用 ROS 进程杀死命令
        "launch_cmd": 'nohup bash -ic "cd ~/kuavo_ros_application && export DISPLAY=:1.0 && source devel/setup.bash && roslaunch dynamic_biped sensor_robot_enable.launch" &',
        "nodes_to_check": [
        "/apriltag_ros_continuous_node",
        "/ar_control_node",
        "/camera/realsense2_camera",
        "/camera/realsense2_camera_manager",
        "/camera_to_real_frame",
        "/joint_state_publisher",
        "/play_music_node",
        "/point_cloud_mask_node",
        "/realsense_yolo_segment_node",
        "/realsense_yolo_transform_torso_node",
        "/record_music_node",
        "/robot_state_publisher",
        "/rosout",
        "/rviz"
        ]
    },
    {
        "name": "lidar",
        "host": "192.168.26.1",
        "port": 22,
        "username": "kuavo",
        "password": "leju_kuavo",
        "remote_path": "/home/kuavo/kuavo_ros_navigation",
        "pkill_cmd": "pkill -f lidar",  # 假设 lidar 特定进程杀死命令
        "launch_cmd": 'nohup bash -ic "cd ~/kuavo_ros_navigation && export DISPLAY=:1.0 && source devel/setup.bash && roslaunch livox_ros_driver2  start_mid360.launch" &',
        "nodes_to_check": [
            "/apriltag_ros_continuous_node",
            "/camera/realsense2_camera",
            "/camera/realsense2_camera_manager",
            "/camera_to_real_frame",
            "/livox_lidar_publisher2",
            "/robot_state_publisher",
            "/rosout",
            "/rviz"
        ]
    }
]

# 定义实时输出函数（线程安全）
# def print_output(stream, prefix=""):
#     for line in iter(stream.readline, ""):
#         print(f"[{prefix}] {line}", end="")
# 定义实时输出函数（线程安全），修改为不打印日志
def print_output(stream, prefix=""):
    for _ in iter(stream.readline, ""):
        pass

def check_nodes(ssh, service_config):
    """通用节点检查函数"""
    try:
        source_command = 'bash -ic "rosnode list"'
        stdin, stdout, stderr = ssh.exec_command(source_command)
        output = stdout.read().decode()
        running_nodes = output.strip().split('\n')
        missing_nodes = [node for node in service_config["nodes_to_check"] if node not in running_nodes]
        if missing_nodes:
            print(f"[{service_config['name']}] 以下节点未启动: {', '.join(missing_nodes)}")
            return False
        print(f"[{service_config['name']}] 所有节点启动成功")
        return True
    except Exception as e:
        print(f"[{service_config['name']}] 节点检查失败: {e}")
        return False

def start_service(service_config):
    """独立启动单个服务的函数"""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # 1. 连接 SSH
        ssh.connect(
            hostname=service_config["host"],
            port=service_config["port"],
            username=service_config["username"],
            password=service_config["password"]
        )

        # 2.1 执行命令检查路径是否存在
        remote_path = service_config["remote_path"]
        command = f"test -e {remote_path} && echo 'Exists' || echo 'Not exists'"
        stdin, stdout, stderr = ssh.exec_command(command)
        # 获取命令结果
        result = stdout.read().decode().strip()
        if result != 'Exists':
            ssh.close()
            print(remote_path + result)
            return None, False

        # 2. 杀死旧进程
        stdin, stdout, stderr = ssh.exec_command(service_config["pkill_cmd"])
        # 可选：打印杀进程输出（根据需要保留或注释）
        # threading.Thread(target=print_output, args=(stdout, f"{service_config['name']}_pkill_stdout")).start()
        # threading.Thread(target=print_output, args=(stderr, f"{service_config['name']}_pkill_stderr")).start()

        # 3. 启动 ROS 程序
        launch_cmd = service_config["launch_cmd"]
        stdin, stdout, stderr = ssh.exec_command(launch_cmd)

        # 4. 启动输出线程
        stdout_thread = threading.Thread(
            target=print_output,
            args=(stdout, service_config["name"].upper())
        )
        stderr_thread = threading.Thread(
            target=print_output,
            args=(stderr, f"{service_config['name'].upper()}_ERROR")
        )
        stdout_thread.start()
        stderr_thread.start()

        # 5. 等待启动并检查节点
        time.sleep(5)  # 给启动留时间
        success = check_nodes(ssh, service_config)

        return ssh, success

    except Exception as e:
        print(f"[{service_config['name']}] 启动失败: {e}")
        ssh.close()
        return None, False

def start_all_services():
    active_ssh_connections = []
    for service in SERVICES:
        print(f"\n==== 启动 {service['name']} 服务 ====")
        ssh, success = start_service(service)
        if not success:
            print(f"{service['name']} 启动失败，终止脚本")
            # 关闭已启动的连接
            for conn in active_ssh_connections:
                conn.close()
            sys.exit(1)
        active_ssh_connections.append(ssh)  # 保存活动连接
    print("\n所有服务启动成功，等待关闭连接...")
    return active_ssh_connections

def close_all_connections(active_ssh_connections):
    print("\n正在关闭所有 SSH 连接...")
    for ssh in active_ssh_connections:
        try:
            stdin, stdout, stderr = ssh.exec_command(pkill_command)
            ssh.close()
        except Exception as e:
            print(f"关闭连接出错: {e}")
    print("所有 SSH 连接已关闭")


if __name__ == "__main__":
    active_ssh_connections = start_all_services()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        close_all_connections(active_ssh_connections)    