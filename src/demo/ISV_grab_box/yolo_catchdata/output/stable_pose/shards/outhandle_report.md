# outhandle 物理稳定姿态报告

状态：**通过**

## 方法

- 工件在机器人面前的真实 `table_box` 中从指定高度释放。
- RGB、Mask 使用原始 USD 可视网格，没有抽稀或减面。
- 动态碰撞体近似：`convexDecomposition`。
- `table_box` 静态碰撞近似：`authored_collision_meshes`。
- 碰撞近似仅影响沉降，不参与 RGB、Mask、Depth 或尺度计算。

## 稳定判定

- 仿真帧数：`179`。
- 连续稳定帧数：`60`。
- 最终线速度：`0.000000 m/s`。
- 最终角速度：`0.000000 °/s`。
- 沉降平移量：`0.065335 m`。
- 沉降旋转变化：`12.848379°`。
- 最终世界包围盒最低点：`1.007424 m`。
- 边界计算方法：`exact_transformed_visual_mesh_vertices`。
- 估计支撑接触：`floor`。
- 工件与侧壁接触并形成静止支撑属于有效自然稳定姿态，不作为失败条件。

## 输出

- 分片：`/home/csc/workspace/projects/kuavo_ws_csc/src/demo/ISV_grab_box/yolo_catchdata/output/stable_pose/shards/outhandle.json`
- 原始 RGB：`/home/csc/workspace/projects/kuavo_ws_csc/src/demo/ISV_grab_box/yolo_catchdata/output/stable_pose/preview/outhandle_settled_rgb.png`
- 二值 Mask：`/home/csc/workspace/projects/kuavo_ws_csc/src/demo/ISV_grab_box/yolo_catchdata/output/stable_pose/preview/outhandle_settled_mask.png`
- 轮廓参考图：`/home/csc/workspace/projects/kuavo_ws_csc/src/demo/ISV_grab_box/yolo_catchdata/output/stable_pose/preview/outhandle_settled_reference.png`
