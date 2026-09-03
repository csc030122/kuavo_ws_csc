# 真实右腕相机最大/最小半径图像报告

状态：**单张方向预览**

## 方法

- 使用实验室场景右侧灵巧手支架上的 `right_wrist_d405_rgb`，没有使用自由观察相机。
- 为避免机器人 articulation 覆盖手部网格变换，采集时引用同一 `zarm_r7_link` 子树建立独立腕部刚体；几何和相机安装外参保持原值。
- 不求解机械臂 IK；将 `zarm_r7_link`、灵巧手和腕部相机整体放置，使相机精确位于指定半径并指向 sampling center。
- `roll=0°` 定义为：在保持光轴不变时，使四指方向最接近世界 `-Y` 的绕光轴角度。
- 本轮按要求不检查机械臂、灵巧手或箱体碰撞，也不进行轨迹规划。
- 工件采用阶段三得到的自然沉降姿态。
- 青色为目标实例 Mask 轮廓，黄色为可见 Mask 包围框。

## 图像指标

| 类别 | 指定半径/m | 实际半径/m | 放置误差/m | 目标射线点积 | Mask 像素 | 可见 bbox 宽度比例 | Mask 深度有效率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| left | 0.50 | 0.5000 | 0.000000 | 1.000000 | 31912 | 0.180 | 1.000 |

## 输出

- 对比总图：`单张预览模式未生成`
- 元数据：`/home/csc/workspace/projects/kuavo_ws_csc/src/demo/ISV_grab_box/yolo_catchdata/output/workspace_calibration/wrist_extremes/right_wrist_radius_extremes_preview.json`
- 单张图像目录：`/home/csc/workspace/projects/kuavo_ws_csc/src/demo/ISV_grab_box/yolo_catchdata/output/workspace_calibration/wrist_extremes/images`
