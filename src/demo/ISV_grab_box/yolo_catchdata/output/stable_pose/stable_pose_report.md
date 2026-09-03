# 阶段三：四类工件物理稳定姿态报告

总体状态：**通过**

## 统一约定

- 所有工件均在真实 `table_box` 中执行重力沉降。
- 工件接触箱底或侧壁并达到静止，均视为有效自然稳定姿态。
- 可视网格为原始 USD Render Mesh，没有抽稀、减面或代理替换。
- 动态碰撞近似：`convexDecomposition`。
- `table_box` 静态碰撞近似：`authored_collision_meshes`。
- 所有运行时物理 API 都写入 Session Layer，源 USD 均未保存或修改。

## 沉降结果

| 类别 | 判稳帧 | 平移量/m | 旋转变化/° | 最终线速度/m·s⁻¹ | 最终角速度/°·s⁻¹ |
|---|---:|---:|---:|---:|---:|
| left | 179 | 0.065093 | 2.764220 | 0.000000 | 0.000000 |
| right | 179 | 0.085462 | 36.502233 | 0.000000 | 0.000000 |
| hose | 179 | 0.074434 | 10.544571 | 0.000000 | 0.000000 |
| outhandle | 179 | 0.065335 | 12.848379 | 0.000000 | 0.000000 |

## 预览

- `left`：`/home/csc/workspace/projects/kuavo_ws_csc/src/demo/ISV_grab_box/yolo_catchdata/output/stable_pose/preview/left_settled_reference.png`
- `right`：`/home/csc/workspace/projects/kuavo_ws_csc/src/demo/ISV_grab_box/yolo_catchdata/output/stable_pose/preview/right_settled_reference.png`
- `hose`：`/home/csc/workspace/projects/kuavo_ws_csc/src/demo/ISV_grab_box/yolo_catchdata/output/stable_pose/preview/hose_settled_reference.png`
- `outhandle`：`/home/csc/workspace/projects/kuavo_ws_csc/src/demo/ISV_grab_box/yolo_catchdata/output/stable_pose/preview/outhandle_settled_reference.png`

## 输出

- 稳定姿态配置：`/home/csc/workspace/projects/kuavo_ws_csc/src/demo/ISV_grab_box/yolo_catchdata/configs/stable_poses.json`
