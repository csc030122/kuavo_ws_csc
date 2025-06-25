.. _kuavo-strategy:
.. include:: components/tips_launch_kuavo.rst

********
策略模块
********
原子技能层级
====================================================
.. currentmodule:: kuavo_humanoid_sdk.kuavo_strategy

.. autoclass:: kuavo_humanoid_sdk.KuavoRobot
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

.. autoclass:: KuavoRobotVision
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:
    
.. autoclass:: KuavoRobotState
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:


.. autoclass:: KuavoRobotTools
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:
    
基础策略接口
====================
.. currentmodule:: kuavo_humanoid_sdk.kuavo_strategy

.. autoclass:: kuavo_humanoid_sdk.kuavo_strategy.kuavo_strategy.KuavoRobotStrategyBase
    :members:
    :undoc-members:
    :show-inheritance:

箱子抓取策略
====================
.. currentmodule:: kuavo_humanoid_sdk.kuavo_strategy.grasp_box

箱子信息数据结构
------------------------
.. autoclass:: kuavo_humanoid_sdk.kuavo_strategy.grasp_box.grasp_box_strategy.BoxInfo
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

箱子抓取策略类
------------------------
.. autoclass:: kuavo_humanoid_sdk.kuavo_strategy.grasp_box.grasp_box_strategy.KuavoGraspBox
    :members:
    :undoc-members:
    :show-inheritance:

搬箱子示例
============

以下是一个使用箱子抓取策略的基本示例:

gazebo 仿真运行
-----------------

准备
^^^^^^^^^^^^

第一次启动 gazebo 场景前需要修改tag尺寸:

在 ``/opt/ros/noetic/share/apriltag_ros/config/tags.yaml`` 文件中将 tag 的 size 尺寸修改为和立方体 tag 码的尺寸一致（只需做一次）

.. code-block:: yaml

    standalone_tags:
      [
        {id: 0, size: 0.088, name: 'tag_0'},
        {id: 1, size: 0.088, name: 'tag_1'}, 
        {id: 2, size: 0.088, name: 'tag_2'},
        {id: 3, size: 0.088, name: 'tag_3'},
        {id: 4, size: 0.088, name: 'tag_4'},
        {id: 5, size: 0.088, name: 'tag_5'},
        {id: 6, size: 0.088, name: 'tag_6'},
        {id: 7, size: 0.088, name: 'tag_7'},
        {id: 8, size: 0.088, name: 'tag_8'},
        {id: 9, size: 0.088, name: 'tag_9'}
      ]

编译
^^^^^^^^^^^^
首先需要编译相关功能包:

.. code-block:: bash

    catkin build humanoid_controllers kuavo_msgs gazebo_sim ar_control
    
运行
^^^^^^^^^^^^

.. warning::
    在运行之前, 需要确认机器人版本 ``ROBOT_VERSION=45`` ，否则会机器人末端控制会有问题

启动仿真环境:

.. code-block:: bash

    source devel/setup.bash

    # 启动gazebo场景
    roslaunch humanoid_controllers load_kuavo_gazebo_manipulate.launch joystick_type:=bt2pro

    # 启动ar_tag转换码操作和virtual操作
    roslaunch ar_control robot_strategies.launch  

.. note::
    每次启动gazebo场景后需要手动打光, 需要在机器人腰部位置附近给个点光源, 否则会找不到 tag, 如下图所示:

    .. image:: ../../docs/images/gazebo.jpg

实物运行
--------------

准备
^^^^^^^^^^^^

上位机需要先修改 ``./src/ros_vision/detection_apriltag/apriltag_ros/config/tags.yaml`` 文件, 将 tag 的 size 尺寸修改为实际大小，比如 0.1 米

.. code-block:: yaml

    standalone_tags:
        [
            {id: 0, size: 0.1, name: 'tag_0'},
            {id: 1, size: 0.1, name: 'tag_1'},
            {id: 2, size: 0.1, name: 'tag_2'},
            {id: 3, size: 0.1, name: 'tag_3'},
            {id: 4, size: 0.1, name: 'tag_4'},
            {id: 5, size: 0.1, name: 'tag_5'},
            {id: 6, size: 0.1, name: 'tag_6'},
            {id: 7, size: 0.1, name: 'tag_7'},
            {id: 8, size: 0.1, name: 'tag_8'},
            {id: 9, size: 0.1, name: 'tag_9'}
        ]

编译
^^^^^^^^^^^^
首先需要编译相关功能包:

.. code-block:: bash

    catkin build humanoid_controllers grab_box

运行
^^^^^^^^^^^^

下位机运行
^^^^^^^^^^^^

.. code-block:: bash

    source devel/setup.bash
    roslaunch humanoid_controllers load_kuavo_real.launch joystick_type:=bt2pro

上位机运行
^^^^^^^^^^^^

.. code-block:: bash

    cd ~/kuavo_ros_application
    source devel/setup.bash
    roslaunch dynamic_biped apriltag.launch

示例代码
--------------
.. literalinclude:: ../../examples/strategies/grasp_box_example.py
  :language: python