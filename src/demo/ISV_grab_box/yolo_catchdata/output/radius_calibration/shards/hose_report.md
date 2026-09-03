# 阶段三：四类工件半径标定报告

总体状态：**通过**

## 标定定义

- `r`：相机光心到工件 sampling center 的距离，单位为米。
- `s_pre_occlusion`：无外部遮挡时，完整目标 Mask 的 bbox 宽度除以图像宽度。
- `truncation_ratio`：未裁剪 CAD AABB 投影框落在图像外的面积比例。
- 标定使用隔离米制 Stage，只加载当前工件，不加载支撑面、机器人或料箱；工件位姿来自真实 `table_box` 物理沉降结果。
- 隔离 Stage 只用于获得无遮挡完整 Mask；源 USD 不保存、不修改。
- 红叉扫描点表示 Mask、Depth 或截断质量不足，只留作边界记录，不参与半径插值。

## 运行环境

- GPU：NVIDIA GeForce RTX 5060 Laptop GPU，驱动 595.84，显存 8151 MiB
- 渲染器：`RaytracedLighting`
- 标定记录数：`63`
- 恢复的地面纹理完整：`True`

## 类别级闭环初值

| 类别 | 可用尺度范围 | Far | Mid | Near | Pre-grasp |
|---|---|---|---|---|---|
| hose | 0.178–0.640 | 不可达 | r0=0.726，搜索=[0.400, 0.900] | r0=0.469，搜索=[0.350, 0.650] | 不可达 |

## 代表视角

- `frontal_mid`：θ=0.0°，φ=50.0°，ψ=0.0°
- `left_low_roll`：θ=-40.0°，φ=35.0°，ψ=-20.0°
- `right_high_roll`：θ=40.0°，φ=65.0°，ψ=20.0°

## 覆盖限制

- hose 在当前 D405 深度范围和代表视角中无法达到 far 目标尺度 0.150。
- hose 的 mid 区间不能在全部代表视角完整覆盖；阶段四应按视角使用具体搜索边界。
- hose 的 near 区间不能在全部代表视角完整覆盖；阶段四应按视角使用具体搜索边界。
- hose 在当前 D405 深度范围和代表视角中无法达到 pre_grasp 目标尺度 0.650。

## 输出

- 标定配置：`/home/csc/workspace/projects/kuavo_ws_csc/src/demo/ISV_grab_box/yolo_catchdata/output/radius_calibration/shards/hose.json`
- 曲线图：`/home/csc/workspace/projects/kuavo_ws_csc/src/demo/ISV_grab_box/yolo_catchdata/output/radius_calibration/shards/hose_curves.png`
- 参考预览：`/home/csc/workspace/projects/kuavo_ws_csc/src/demo/ISV_grab_box/yolo_catchdata/output/radius_calibration/preview/hose_scale_reference.png`
