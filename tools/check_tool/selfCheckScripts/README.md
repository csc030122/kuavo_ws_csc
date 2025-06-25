一键自检脚本使用说明：
1. 跳转到tools/check_tool/selfCheckScripts路径下
    cd {project_path}/tools/check_tool/selfCheckScripts

2. 启动脚本(在root权限下运行)
    python3 oneKeySelfCheck.py

3. 下位机自检

    在ros-control中编译相关库
    catkin build hardware_node 

    在opensource中要编译
    catkin build humanoid_controllers 

    source devel/setup.bash

    运行oneKeySelfCheck.py脚本选择 1

    脚本启动后自动对IMU，灵巧手，手部和腿部电机进行检测。终端会打印一些关键信息，如果终端打印的日志有异常，详细请查看同级目录下roslaunch_output.log文件

    ps:下位机自检比较危险，程序在硬件自检完成后会对分别对手部电机和腿电机出进行小幅度转动，用户需要随时关注手部电机运动是否发生碰撞(如果碰撞大概率是电机零位不对引起的)，如果出现异常，关闭程序或按急停，避免碰撞产生的长时间堵转！！！之后腿部电机会进入标定姿态，请在自检前将机器人挂高，预留足够的腿部空间！！！


 
4. 上位机自检
    直接运行脚本，选择 2 ，失败可以多次尝试运行（如果节点无论怎么样都启动失败，请查看上位机是否相关工程文件）

    脚本启动后会自动启动上位机的ros_application和ros_navigation程序,测试音响，麦克风，RGB和Depth图片获取，以及雷达信号获取

5. h12信号检测
    运行脚本，选择 3 

6. 电机跟随性测试(2025.06.06新增)
    (1). 运行方式(机器初始化会进入标定姿态，需要将机器人挂高，后续执行会动脚掌，需要给脚掌预留运动空间)
    ros-control中需依赖hardware_node包
    opensource中需要依赖humanoid_controllers包

    运行脚本，选择 4 

    （2）.生成图像
    在selfcheckScripts/motorTest路径中
    cd motorTest
    python3 drawWaveform.py

    生成的图像在/image文件夹中

6. 如果在脚本运行中遇到无法解决问题，请联系唐宁进行反馈和解决
