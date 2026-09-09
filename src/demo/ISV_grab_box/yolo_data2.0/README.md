# YOLO 左右腕侧置相机数据采集器

本目录只保留当前使用的 S63 左/右腕侧置 D405 方案，为 `left`、`right`、
`hose`、`outhandle` 四类工件生成 RGB、Z-depth、实例 Mask 和相机坐标系下的 6D 位姿。
两只手使用相同随机化分布，但必须分别生成计划、manifest、配额和输出目录，
不在同一批次中同时渲染。相同 `--seed` 下右手保持原随机序列，左手使用另一条可重现
序列，避免两分支生成完全重复的样本。
旧的悬空/手背上方相机半径扫描、左右腕统一半径表和 `theta/phi` look-at 方案已经移除，
不得再用于正式数据。

## 唯一有效场景与相机

- 场景：`yolo_scene/lab_yolo_grab_data_s63_mount_corrected.usd`
- 箱体：`/World/Environment/Geometry/box/table_box`
- 相机：所选手侧的侧置 D405，刚性跟随手掌，不执行动态 look-at
- 分辨率和针孔参数：`configs/camera.yaml`
- 四类工件及 canonical frame：`configs/assets.yaml`
- 正式随机化、沉降和质量门禁：`configs/collection.yaml`

修正场景通过子层引用基础实验室 USD，并覆盖错误的手腕安装旋转；基础 USD 只是该修正
场景的内部依赖，不是可选采集入口。

## 当前采集链路

```text
按 class × distance_stage × split 统计缺额
→ 生成确定性随机计划
→ 工件在 table_box 中心安全域低高度释放
→ PhysX 落到箱底并冻结
→ 根据实际沉降平面朝向传递右手基准或左手镜像 grasp 参考
→ 沿工件中心到侧置相机的视线构造接近距离层
→ 加接近轨迹横向和姿态扰动
→ 相机退出箱内平面后，整只手上抬到视线越过箱沿
→ 侧置 D405 渲染 RGB/Depth/Mask/T_C_O
→ 可见率与深度门禁
→ 失败时定向重试一次
→ 写入 split 和 manifest
```

### 工件放置

- `left/right/hose` 只在箱体长轴的 `0°/180°` 附近采样小范围 yaw，避免长工件横放后
  必须依靠侧壁才能进入箱体。
- 支撑面与 yaw 带使用独立分层索引，候选会覆盖 `+z/-z × 0°/180°` 四种组合，
  不再把 `+z` 固定绑定到 `0°`、把 `-z` 固定绑定到 `180°`。
- `outhandle` 尺寸较小，保留 `0–360°` yaw。
- 所有类别只在箱体中心安全域采样 XY；释放时至少预留 2 cm，沉降后要求与侧壁至少
  保留约 1 cm。
- 释放高度为 2 cm。请求支撑面只决定初始姿态；只要最终静止、接触箱底且不靠侧壁，
  翻到其它支撑面也保留。

### 接近轨迹

每个类别按 `grasp → pre_grasp → near → approach → far` 五层采集。前三类距离分别为
`0`、`0.02–0.10`、`0.10–0.20`、`0.20–0.35`、`0.35–0.50 m`；`outhandle` 为
`0`、`0.01–0.06`、`0.06–0.12`、`0.12–0.18`、`0.18–0.25 m`。

`grasp` 保留标定终点。其余距离层把整只手连同真实固定外参的 D405 沿“工件中心→相机”
射线后退。当相机的 XY 位置退出 `table_box` 内壁时，会根据实际工件中心、相机位置和
箱沿高度计算最小竖直抬升量，使工件—相机视线越过箱沿。这一步移动整只手，不改变掌到
相机的固定外参，也不旋转或单独瞄准相机。长工件的手掌平面 yaw 使用实际沉降后的方向，
而不是仅使用释放前请求的 yaw；roll/pitch 仍保持标定姿态。

手掌横向/姿态扰动随距离收缩：far 为 `±7 cm/±10°`，approach 为 `±5 cm/±8°`，
near 为 `±3 cm/±5°`，pre-grasp 为 `±1.5 cm/±3°`，grasp 不扰动。各层以中心分布
为主，仍保留 `10%–25%` 的全范围尾部来覆盖轨迹边界。相机安装外参和手指参考角不随机。

## Mask 与深度有效条件

目标单独宽视场真实 Mesh 投影提供“完整工件”的参考面积。
`visibility_reference.visible_fraction_of_full_mask` 只表示工件投影落在画面内的理论比例，
用来区分出画；正式门禁使用
`visibility_reference.scene_visible_fraction_of_full_mask`，即“真实场景实例 Mask 像素数 /
换算到真实相机焦距的完整工件参考面积”。因此出画、手部遮挡和 `table_box` 遮挡都会
降低有效可见率。当前标准为：

- 目标实例和 Mask 必须非空；
- 可见比例目标为 50%；默认硬下限为 45%，`hose` 放宽到 40%。现有样本复核表明，
  `hose` 在 40%–45% 时仍有清晰、面积足够的轮廓，而约 29% 的样本确实裁切过多；
- 不再使用 `5000 px` 严格门槛，只保留 `500 px` 安全底线以过滤肉眼仅为小点的样本；
- 可见 Mask 中默认至少 50% 像素具有有效 Z-depth。`hose/grasp` 因侧置 D405 的
  `7 cm` 近裁剪面放宽到 30%；这些样本仍约有 8 万个有效深度像素；
- 8 倍宽视场参考自身触边时说明“完整面积”分母仍不可信，直接拒绝并在重试时沿相机
射线后退；即使参考触边，对应类别的可见率硬下限也始终执行；
- 只是出画时重试会后退并收缩扰动；画内但被箱沿/手遮挡时，重试会后退以抬高视线，
同时收缩扰动。

## 右手基准与左手自动镜像

四类 `grasp/pregrasp` 只需在右手分支人工标定：

```bash
/home/csc/isaacsim/python.sh scripts/author_approach_reference.py \
  --hand-side right --class-name right
```

左手参考由右手基准自动生成；不是简单把一个坐标取反，而是同时使用 S63 模型中的
左右 `r7 → palm` 坐标系约定和左腕侧置 D405 的真实固定外参：

```bash
python3 scripts/mirror_approach_references.py
```

运行左手批采或配额采集时也会自动执行这一步；右手参考一旦变化，已有自动镜像文件会按
源文件 SHA-256 自动刷新。若以后确实保存了完整的手工左手参考，默认会保留，不会被自动
覆盖；只有显式传入 `--force` 才覆盖。

输出位于：

```text
output/manual_wrist/references/<hand_side>_wrist_<class>_approach_reference.json
```

镜像平面取经过每份参考工件原点、法向为世界 X 的平面，因此保留右手的手—物距离和
`grasp → pregrasp` 轨迹长度，并把横向接近方向翻到左侧。两侧随机化范围相同且关于零
对称，但随机计划和输出仍是两个独立分支。标定界面中的 `D405 View` 是当前手侧的真实
侧置相机视图；相机不会被单独旋转去对准工件。

## 左右手连续预采集

```bash
python3 scripts/collect_bimanual_approach_pilot.py \
  --output-root output/dataset/approach_open_pilot_bimanual \
  --accepted-per-class-stage 5 \
  --samples-per-class-per-round 25 \
  --max-rounds 4 \
  --seed 20260908 \
  --isaac-python /home/csc/isaacsim/python.sh
```

一条命令先采右手，再继续采左手。默认每个相机收集
`4 类 × 5 轨迹层 × 5 = 100` 个通过样本，左右合计 200 个。同一沉降工件的五个
轨迹层始终放在同一 split，避免泄漏。配额按 `class × 轨迹层 × split` 追踪；由于每格
5 张无法单独精确表达 70/15/15，整数余数会在 20 个格之间轮转，使每个相机最终总数
精确为 `train/val/test = 70/15/15`。

输出结构：

```text
approach_open_pilot_bimanual/
├── right_wrist_d405/
│   ├── train/   # 仅 <class>_<id>_{raw.png,depth.npy,mask.png,pose.json}
│   ├── val/
│   ├── test/
│   ├── _metadata/
│   ├── rejected/
│   └── manifest.jsonl
├── left_wrist_d405/
│   └── ...
└── pilot_run_summary.json
```

`train/val/test` 中只有用户要求的四件套；质量指标放在 `_metadata`，失败图像放在
`rejected/<split>`。`pose.json` 中 `R_vec={a,b,c}`、`t_vec={x,y,z}`，并附带可核对的
`R_matrix` 和 `camera_from_object`。每条 manifest 记录工件落点、实际支撑面、手掌扰动、
质量指标和重试原因。

## 左右手独立配额采集

```bash
python3 scripts/collect_approach_until_quota.py \
  --hand-side right \
  --output-root output/dataset/approach_open_formal/right_hand \
  --samples-per-class-per-round 25 \
  --accepted-per-class-stage 20 \
  --max-rounds 8 \
  --seed 20260908 \
  --isaac-python /home/csc/isaacsim/python.sh \
  --strict

python3 scripts/collect_approach_until_quota.py \
  --hand-side left \
  --output-root output/dataset/approach_open_formal/left_hand \
  --samples-per-class-per-round 25 \
  --accepted-per-class-stage 20 \
  --max-rounds 8 \
  --seed 20260908 \
  --isaac-python /home/csc/isaacsim/python.sh \
  --strict
```

每只手的目标为 `4 类 × 5 层 × 20 = 400` 个通过样本，两分支合计 800 个。
每轮只补未满的 `class × 轨迹层 × split` 格；同一 scene group
共享一次工件沉降结果和同一 train/val/test split。Isaac 包装进程若未产出沉降分片，
会最多重新启动两次。

采集器会在输出根目录持有 `.collection.lock`。若已有采集器写同一个目录，新进程会立即
报错退出，避免两个 manifest/编号生成器同时写入导致 ID 冲突和跨 split 泄漏。锁文件可
保留；真正的锁由内核维护，进程退出后会自动释放。

若旧数据曾被并发写入，或希望按新门禁重新纳入原先误拒的样本，可无损生成干净目录：

```bash
/home/csc/isaacsim/python.sh scripts/sanitize_approach_dataset.py \
  --source-root output/dataset/approach_open_formal_v2/right_wrist_d405 \
  --output-root output/dataset/approach_open_formal_v3/right_wrist_d405 \
  --accepted-per-class-stage 20
```

清洗器不修改源目录；它重新执行当前质量门禁、恢复符合新标准的 rejected 样本、过滤小于
500 px 的 Mask、去除完全重复轨迹，并保证同一沉降 scene group 只进入一个 split。

## 验证

```bash
cd /home/csc/workspace/projects/kuavo_ws_csc/src/demo/ISV_grab_box/yolo_catchdata
PYTHONPATH=. python3 -m unittest discover -s tests -v
python3 scripts/collect_approach_until_quota.py \
  --hand-side right \
  --output-root /tmp/approach_quota_dryrun \
  --dry-run
```

6D 位姿保存为 `T_C_O`，即 canonical object frame 到 OpenCV 相机系的变换；OpenCV
约定为 `+X` 向右、`+Y` 向下、`+Z` 向前，旋转同时保存矩阵和 Rodrigues 向量。
