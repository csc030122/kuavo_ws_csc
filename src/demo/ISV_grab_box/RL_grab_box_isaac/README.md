# Kuavo S63 三指抓取 Stage 0

该目录是后续 Isaac Lab 强化学习抓取任务的独立第一阶段资产与基准场景。当前只验证资产、坐标、碰撞、重力和关节控制，不包含 PPO、reward、observation、相机、料箱或并行环境。

## 场景范围

- 保留完整 Kuavo S63 几何与真实右臂安装关系，robot root 固定。
- 仅保留 13 个物理 DOF：右臂 7 DOF，以及右手拇指、食指、中指各 2 DOF。
- 左臂、腿、腰、头、ring 和 little finger 的 body/visual/collision 仍在，但通过删除其 MJCF joint 后作为 fixed chain 导入。
- 桌面上表面为 `z=1.000 m`，透明临时支撑面上表面为 `z=1.010 m`。
- 桌面中心为 `(0.72, 0.0, 0.95)`，位于机器人 `+X` 正前方中线；桌面前后尺寸缩为 `0.64 m`，近边缘 `x=0.40 m`，给前轮与桌腿留出间隙。
- 临时支撑板与工件 spawn 的 XY 都是 `(0.72, 0.0)`，即桌面几何中心。
- 外拉手从支撑面上方 `0.020 m` 自由落下；场景中没有料箱。
- 场景使用浅蓝灰 Dome 环境光、暖色主光和浅灰地面，避免默认黑色背景造成低可见度。

## 文件

```text
assets/robots/kuavo_s63_dexhand_rl.xml       # 从 fixed S63 派生的独立 13-DOF MJCF
assets/robots/generated/...                  # MJCF importer 生成的独立 USD package
assets/objects/outer_handle_stage0.usd       # 外拉手 visual/rigid body/convex collision
assets/environment/stage0_table_support.usd  # 地面、桌子和 10 mm 临时支撑层
scenes/kuavo_grasp_stage0.usd                # Stage 0 基准场景
config/stage0_nominal.json                   # 可复用 nominal state 和关键坐标
config/stage0_validation.json                # 最近一次 smoke test 的实测结果
scripts/build_stage0_scene.py                # 可重复生成资产和场景
scripts/test_stage0_robot.py                 # 13 DOF、工件和场景 headless smoke test
```

### 右机械臂独立输出

已另行输出只包含右机械臂与右侧三指灵巧手的版本。该版本从
`zarm_r1_link` 子树直接抽取，不包含底盘、腿、腰、躯干、头或左臂；
无名指和小指仅保留随掌固定的几何，不增加自由度。

```text
assets/robots/kuavo_s63_right_arm_dexhand_rl.xml
assets/robots/generated/kuavo_s63_right_arm_dexhand_rl/
scenes/kuavo_grasp_arm_only.usd
config/arm_only_nominal.json
config/arm_only_validation.json
scripts/build_arm_only_scene.py
```

机械臂独立场景把原整机的右臂安装 frame 设为世界原点，坐标轴方向不变。
桌子、支撑板和工件统一平移 `(-0.11075, 0.253, -1.19601) m`，因此抓取相关
相对位姿完全不变。当前 USD 中工件的释放位置为
`(0.60925, 0.253, -0.14888107) m`、quaternion wxyz
`(0.94906819, -0.29321110, -0.07567815, 0.08700369)`。自然落稳后的最新抓取基准
为 position `(0.61058050, 0.25285539, -0.16413184) m`、quaternion wxyz
`(0.95198900, -0.28714580, -0.04609212, 0.09560210)`。

构建机械臂独立版本：

```bash
/home/csc/isaacsim/python.sh \
  src/demo/ISV_grab_box/RL_grab_box_isaac/scripts/build_arm_only_scene.py
```

验证机械臂独立版本：

```bash
/home/csc/isaacsim/python.sh \
  src/demo/ISV_grab_box/RL_grab_box_isaac/scripts/test_stage0_robot.py \
  --usd src/demo/ISV_grab_box/RL_grab_box_isaac/scenes/kuavo_grasp_arm_only.usd \
  --nominal-json src/demo/ISV_grab_box/RL_grab_box_isaac/config/arm_only_nominal.json \
  --validation-json src/demo/ISV_grab_box/RL_grab_box_isaac/config/arm_only_validation.json \
  --joint-steps 90 --settle-steps 480
```

原始 `tpyrced_数模/外拉手.stp` 和已有的 `tpyrced_数模/外拉手.usd` 仅作为源文件读取，不被覆盖。

## 资产来源与改造

机器人来源是：

```text
src/demo/ISV_grab_box/isaac_sim/biped_s63_Dhand_fixed.xml
```

该文件中的 Dhand 已保持 `zarm_r7_link -> palm -> finger links` 安装链，但 6 个目标 finger joint 被 XML 注释。生成脚本只在派生副本中解除这些注释，并用下列文件核对 joint origin、axis、range、actuator 和 sensor：

```text
src/kuavo_assets/models/biped_s49/xml/biped_s49.xml
src/kuavo_assets/models/biped_s49/urdf/biped_s49.urdf
```

右臂 stage drive gains 取自 `src/kuavo_assets/config/kuavo_v63/kuavo.json` 的右臂 `ruiwo_kp/ruiwo_kd`。S49 MJCF 只给出了 position actuator、joint damping 和 ctrl range，没有物理伺服刚度；因此 finger stage gains 是 Stage 0 验证值，后续做 sim-to-real 前应重新标定。

原始搬箱 Demo 和原始 fixed XML 均不会被生成脚本修改。

## 13 个 active joints

```text
zarm_r1_joint
zarm_r2_joint
zarm_r3_joint
zarm_r4_joint
zarm_r5_joint
zarm_r6_joint
zarm_r7_joint
r_thumbCMC
r_thumbMCP
r_indexMCP
r_indexPIP
r_middleMCP
r_middlePIP
```

PhysX 内部 DOF 顺序可能把同层的 MCP joints 排在 distal PIP joint 之前。控制代码必须按 `robot.dof_names` 建 name-to-index 映射，不能假设 JSON 列表顺序等于 PhysX 顺序。

`r_indexPIP`和`r_middlePIP`的物理joint limit与actuator control range均统一为
`[0, 1.25] rad`（约`[0, 71.62]°`），以S49 URDF硬限位为准。S49参考MJCF中的
`2.0944 rad`只存在于joint range，与其自身actuator的`1.25 rad`不一致，因此不沿用。

## 构建

在仓库根目录运行：

```bash
/home/csc/isaacsim/python.sh \
  src/demo/ISV_grab_box/RL_grab_box_isaac/scripts/build_stage0_scene.py
```

首次运行会生成派生 MJCF 并执行 MJCF→USD 导入。机器人 package 已存在时，可只重建工件、环境、场景和 nominal JSON：

```bash
/home/csc/isaacsim/python.sh \
  src/demo/ISV_grab_box/RL_grab_box_isaac/scripts/build_stage0_scene.py \
  --skip-robot-import
```

若服务器 Isaac Sim 安装路径不同，将 `/home/csc/isaacsim/python.sh` 替换为容器内实际的 Isaac Sim `python.sh`。

## Smoke test

```bash
/home/csc/isaacsim/python.sh \
  src/demo/ISV_grab_box/RL_grab_box_isaac/scripts/test_stage0_robot.py \
  --joint-steps 90 \
  --settle-steps 480
```

测试会：

1. 静态检查场景中恰好存在目标 13 个 revolute joints，ring/little 没有可动 joint，并检查三指碰撞体。
2. 固定 120 Hz CPU PhysX，逐个发送小幅正向 position command，并验证目标关节运动、其他 DOF 扰动和数值有限性。
3. 检查 robot root drift、palm 相对 wrist attachment drift。
4. 验证工件从 2 cm gap 受重力落下、没有穿过支撑/桌面，并在 480 steps 后静止。
5. 验证桌面/支撑高度和 palm-object nominal 距离，写出 `config/stage0_validation.json`。

当前实测结果为 `SMOKE_RESULT=PASS`：13/13 关节通过；工件从桌面中心上方下降 `0.019987 m`，settled center 约为 `(0.720064, -0.000005, 1.027141) m`，最终线速度和角速度均为 `0`；palm-object 距离为 `0.299111 m`；palm attachment drift 为 `4.84e-8 m`；robot root drift 为 `0`。

## Nominal frame 与坐标

固定参考 frame 为：

```text
/World/Robot/ArmBaseReference
```

它位于 `zarm_r1_joint` 的固定父 frame，world pose 为 position `(0.11075, -0.253, 1.19601)`、quaternion wxyz `(1,0,0,0)`。以后拆出 arm-only 资产时，可把该 frame 放到 world origin，并保持 `config/stage0_nominal.json` 中的 `arm_base_to_object_nominal` 不变。

完整 nominal 数值以 `config/stage0_nominal.json` 为准。工件自然落稳后的实测 pose 已直接保存为 `object_nominal_settled_world_pose` 和 `arm_base_to_object_nominal`，后续抓取、拆分 arm-only 资产和强化学习都以该稳定 pose 为基准。Smoke test 会检查每次落稳结果相对该基准的位置误差不超过 5 mm、姿态误差不超过 2 度。最近一次实测的 13 个关节位移和 drift 记录在 `config/stage0_validation.json`。

## 已知限制

- 外拉手质量 `0.18 kg` 是工程估计，进入 sim-to-real 前必须称重并更新；摩擦也需要实物标定。
- 动态 collision 使用 `convexDecomposition`，比直接三角网格稳定，但仍应在高并行训练前制作更低面数的专用 convex hull 集合。
- 当前本地运行环境没有可用 CUDA/RTX，验证使用 CPU PhysX；需要在训练服务器容器中再跑一次同一 smoke test。
- importer 的 articulation self-collision 默认关闭；三指对工件的 collider 有效，但后续若奖励或动作可能造成手指互穿，应评估开启 self-collision 或增加专用碰撞过滤/限位。
- finger drive gains、工件质量和摩擦不是最终训练/实机参数。
- 本阶段故意没有 Isaac Lab task、PPO、reward、observation、相机、料箱和环境克隆。
