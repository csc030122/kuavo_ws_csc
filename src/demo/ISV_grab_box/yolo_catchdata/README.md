# YOLO 腕部相机数据采集器

本项目使用 Isaac Sim 为四类工件生成合成数据：`left`、`right`、`hose` 和
`outhandle`。第一里程碑是完成阶段 0–3 验证，并生成 400 张 Smoke 数据。
`yolo_scene/` 下的现有 USD 均视为只读源资产，采集器不会修改它们。

## 阶段 0–1：项目、资产与坐标系

运行不依赖 Isaac Sim 的配置与坐标系单元测试：

```bash
cd /home/csc/workspace/projects/kuavo_ws_csc/src/demo/ISV_grab_box/yolo_catchdata
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

使用 Isaac Sim Python 执行只读 USD 验证：

```bash
/home/csc/isaacsim/python.sh scripts/verify_assets.py
```

该命令生成：

```text
output/asset_validation/asset_report.json
output/asset_validation/asset_report.md
output/asset_validation/preview/canonical_frames_and_bounds.png
output/asset_validation/preview/table_box_static_placement.png
```

两张预览图由 CPU 绘制，用于检查 canonical frame、sampling center、物理尺寸和
工件静态放置是否越出 `table_box`。阶段二的 RTX 验证会额外检查实际场景渲染、
相机标注数据和投影关系。

## 阶段二：相机、深度与位姿链路

先运行全部纯几何测试，再用可访问 NVIDIA GPU 的终端启动 Isaac Sim：

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
/home/csc/isaacsim/python.sh scripts/verify_pose_convention.py
```

阶段二验证以下约定：

- OpenCV 光学系为 +X 向图像右侧、+Y 向图像下方、+Z 向相机前方；
- 正前方 1 米的目标位移为 `[0, 0, 1]`；
- RGB、二值 mask 和 `distance_to_image_plane` Z-depth 尺寸一致；
- mask 内深度有效，并且 CAD AABB 投影能够包住实例 mask；
- 保存的物体位姿为 `T_C_O`，满足 `T_C_O = T_C_W · T_W_A · T_A_O`。

阶段二输出：

```text
output/pose_validation/pose_report.json
output/pose_validation/pose_report.md
output/pose_validation/preview/rgb.png
output/pose_validation/preview/mask.png
output/pose_validation/preview/depth.png
output/pose_validation/preview/projection_overlay.png
output/pose_validation/sample/depth.npy
```

当前 `configs/camera.yaml` 使用 D405 公开规格中的标称视场角作为合成针孔模型。
真机采集前必须用实际右腕 D405 的内参和畸变标定结果替换，不能把标称值当作真机
标定值。

## 阶段三：自然稳定姿态与腕部观察工作空间

阶段三先在真实实验室场景的 `table_box` 中分别释放四类工件。RGB、Mask 和 Depth
始终使用原始 USD 可视网格；`convexDecomposition` 仅作为动态碰撞近似，不会替换
渲染几何。穿透检查直接使用变换后的实际可视 Mesh 顶点，不使用旋转 AABB 的空角点。
工件与箱底或侧壁接触并达到静止，均属于有效自然稳定姿态。

每个类别使用独立 Isaac Sim 进程生成稳定姿态：

```bash
/home/csc/isaacsim/python.sh scripts/collect_stable_pose.py --class-name left
/home/csc/isaacsim/python.sh scripts/collect_stable_pose.py --class-name right
/home/csc/isaacsim/python.sh scripts/collect_stable_pose.py --class-name hose
/home/csc/isaacsim/python.sh scripts/collect_stable_pose.py --class-name outhandle
/home/csc/isaacsim/python.sh scripts/merge_stable_poses.py
```

稳定姿态输出：

```text
configs/stable_poses.json
output/stable_pose/stable_pose_report.md
output/stable_pose/shards/
output/stable_pose/preview/
```

采样策略采用“抓取接近工作空间优先”：`r` 是相机光心到工件 sampling center 的
距离，是主采样变量；`s_pre_occlusion` 是完整目标二值 Mask 的 bbox 宽度除以图像
宽度，只作为渲染结果和分布诊断量，不再反求 `r`。

当前全局扫描范围为 `0.12–0.50 m`，使用以下九个半径层：

```text
0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.12 m
```

“明显出画”定义为未裁剪 CAD 投影面积在图像外的比例超过 `20%`；近距失效定义为
最近表面 Z-depth 小于 `0.07 m`。中心视角观测表明，`left`、`right`、`hose` 的
有效下界约为 `0.30 m`，`outhandle` 可到 `0.12 m`。全局下界保留 `0.12 m`，后续
通过每类、每个半径层的角度有效域 `Omega_class(r)` 自动排除大工件的过近位置。

生成半径选择依据和中文报告：

```bash
python3 scripts/select_radius_layers.py
```

输出：

```text
configs/workspace_calibration.json
output/workspace_calibration/radius_selection_report.md
```

下一步将在每个半径层扫描候选 `theta/phi` 网格，记录目标可见性、Depth、截断率和
`s_pre_occlusion`，最终保存不规则有效格点集合，而不是简单的全局角度矩形。

之前生成的尺度—半径曲线保留为历史诊断结果：

```bash
/home/csc/isaacsim/python.sh scripts/calibrate_radius.py --class-name left
/home/csc/isaacsim/python.sh scripts/calibrate_radius.py --class-name right
/home/csc/isaacsim/python.sh scripts/calibrate_radius.py --class-name hose
/home/csc/isaacsim/python.sh scripts/calibrate_radius.py --class-name outhandle
/home/csc/isaacsim/python.sh scripts/merge_radius_calibration.py
```

历史诊断输出：

```text
configs/radius_calibration.json
output/radius_calibration/radius_calibration_report.md
output/radius_calibration/radius_scale_curves.png
output/radius_calibration/shards/
output/radius_calibration/preview/
```

这些输出不再驱动正式采样，`scale_solver.py` 也不进入新的单样本生成状态机。

## 固定类别编号

| 编号 | 类别 |
|---:|---|
| 0 | left |
| 1 | right |
| 2 | hose |
| 3 | outhandle |

这些编号保持了旧版工件运行清单的顺序，同时与当前 USD 默认 Prim 名称一致。

四个工件源 USD 使用毫米单位（`metersPerUnit=0.001`），实验室场景使用米单位
（`metersPerUnit=1.0`）。`asset_loader` 会在运行时实例根节点写入统一的 `0.001`
缩放，不会修改源资产图层。
