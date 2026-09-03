# 阶段二：相机、深度与位姿验证报告

总体状态：**通过**

## 运行环境

- GPU：NVIDIA GeForce RTX 5060 Laptop GPU，驱动 595.84，显存 8151 MiB
- 渲染器：`RaytracedLighting`
- 分辨率：`1280 × 720`

## 坐标系约定

- 输出光学坐标系采用 OpenCV：+X 向图像右侧、+Y 向图像下方、+Z 向前。
- 位姿链为 `T_C_O = T_C_W · T_W_A · T_A_O`。
- Depth 来自 `distance_to_image_plane`，含义为 Z-depth，单位为米。

## 验收项

| 验收项 | 状态 | 实际值 | 期望 |
|---|---|---|---|
| `front_one_meter` | 通过 | `[0.0, 0.0, 1.0]` | 正前方 1 米时 t=[0, 0, 1] |
| `image_right_positive_x` | 通过 | `[0.1, 0.0, 1.0]` | 图像右侧 x>0 |
| `image_down_positive_y` | 通过 | `[0.0, 0.1, 1.0]` | 图像下侧 y>0 |
| `nvidia_gpu_available` | 通过 | `{"available": true, "gpus": [{"name": "NVIDIA GeForce RTX 5060 Laptop GPU", "driver_version": "595.84", "memory_total_mib": 8151}]}` | 检测到可用 NVIDIA GPU |
| `rgb_shape` | 通过 | `[720, 1280, 4]` | 前两维为 [720, 1280] |
| `mask_shape` | 通过 | `[720, 1280]` | 尺寸为 [720, 1280] |
| `depth_shape` | 通过 | `[720, 1280]` | 尺寸为 [720, 1280] |
| `camera_intrinsics_authoring` | 通过 | `{"fx": 710.7919877738743, "fy": 649.4572119390921}` | USD Camera 的 fx/fy 与配置计算值误差不超过 1e-3 像素 |
| `target_instance_found` | 通过 | `[2]` | 实例语义包含 hose |
| `mask_pixels` | 通过 | `30927` | >= 1000 |
| `valid_depth_ratio` | 通过 | `1.0` | >= 0.95 |
| `mask_inside_projected_bbox` | 通过 | `1.0` | >= 0.95 |
| `look_at_pose` | 通过 | `[0.0, 5.551115123125783e-17, 0.55]` | 目标中心位于相机前方且在光轴上 |

## 相机内参说明

当前内参来源：`nominal_d405_fov_pending_real_calibration`。这是用于合成链路验证的 D405 标称视场角针孔模型，真机采集前必须替换为实际右腕相机的标定结果。

## 预览与样本文件

- `rgb`：`/home/csc/workspace/projects/kuavo_ws_csc/src/demo/ISV_grab_box/yolo_catchdata/output/pose_validation/preview/rgb.png`
- `mask`：`/home/csc/workspace/projects/kuavo_ws_csc/src/demo/ISV_grab_box/yolo_catchdata/output/pose_validation/preview/mask.png`
- `depth_preview`：`/home/csc/workspace/projects/kuavo_ws_csc/src/demo/ISV_grab_box/yolo_catchdata/output/pose_validation/preview/depth.png`
- `projection_overlay`：`/home/csc/workspace/projects/kuavo_ws_csc/src/demo/ISV_grab_box/yolo_catchdata/output/pose_validation/preview/projection_overlay.png`
- `depth_npy`：`/home/csc/workspace/projects/kuavo_ws_csc/src/demo/ISV_grab_box/yolo_catchdata/output/pose_validation/sample/depth.npy`
- `pose_json`：`/home/csc/workspace/projects/kuavo_ws_csc/src/demo/ISV_grab_box/yolo_catchdata/output/pose_validation/sample/pose.json`
