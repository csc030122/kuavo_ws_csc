# 双 GPU 正式采集迁移包

本目录是可独立上传的最小完整采集包，不包含本机已有数据集，也不依赖原仓库的其它目录。
它包含当前验证通过的：

- S63 左右腕侧置 D405 场景和四类工件 USD；
- 左右手抓取/预抓取参考；
- table_box 物理沉降、接近轨迹、箱沿视线抬升和质量门禁代码；
- RGB、Z-depth、二值 Mask、相机坐标系 6D 位姿写出链路；
- 双 GPU 并行、停止、状态查看和断点续采脚本；
- 回归测试。

从当前机器上传整个普通文件夹（任选一种）：

```bash
rsync -a --info=progress2 \
  /home/csc/workspace/yolo_catchdata_server_dual_gpu_5000 \
  用户名@服务器:/上传路径/
```

```bash
scp -r /home/csc/workspace/yolo_catchdata_server_dual_gpu_5000 \
  用户名@服务器:/上传路径/
```

## 服务器要求

- Linux、两张可被 `nvidia-smi` 识别的 NVIDIA GPU；
- Isaac Sim 6.0（默认 `/home/csc/isaacsim/python.sh`）；
- 系统 `python3` 和 PyYAML；
- 建议至少 100 GiB 可用磁盘。当前预采实测约 5 MB/通过样本，加上 rejected、
  元数据和中间结果，左右手各 5000 张应按 60 GiB 左右预留，100 GiB 更安全。

若 Isaac 安装路径不同：

```bash
export ISAAC_PYTHON=/实际路径/isaacsim/python.sh
```

## 上传后检查

```bash
cd /上传路径/yolo_catchdata_server_dual_gpu_5000
chmod +x run_dual_gpu_5000.sh stop_dual_gpu.sh status_dual_gpu.sh \
  server_tools/preflight_server.sh
./server_tools/preflight_server.sh
```

预检会核对两张 GPU、必要 USD、八份左右手参考、SHA-256、磁盘空间、89 项测试和
两侧 dry-run。必须看到 `PREFLIGHT=PASS`。

## 启动

默认映射为右手 GPU 0、左手 GPU 1：

```bash
cd /上传路径/yolo_catchdata_server_dual_gpu_5000
mkdir -p output/logs output/run_state
nohup ./run_dual_gpu_5000.sh \
  > output/logs/dual_gpu_launcher.log 2>&1 < /dev/null &
echo $! > output/run_state/launcher.pid
```

如果希望交换显卡：

```bash
RIGHT_GPU=1 LEFT_GPU=0 nohup ./run_dual_gpu_5000.sh \
  > output/logs/dual_gpu_launcher.log 2>&1 < /dev/null &
echo $! > output/run_state/launcher.pid
```

每只手目标为 `4 类 × 5 轨迹层 × 250 = 5000` 张，单手 split 为
`train/val/test = 3500/750/750`。两只手并行，但使用独立计划、稳定姿态命名空间、
输出目录、manifest 和内核文件锁。

输出位于：

```text
output/dataset/approach_open_formal_5000_each/
├── right_wrist_d405/
│   ├── train/
│   ├── val/
│   ├── test/
│   ├── _metadata/
│   ├── rejected/
│   ├── plans/
│   └── manifest.jsonl
├── left_wrist_d405/
│   └── ...
├── logs/
└── run_state/
```

## 查看、停止和续采

```bash
./status_dual_gpu.sh
./stop_dual_gpu.sh
```

停止或服务器重启后，用同一条启动命令、同一个 `OUTPUT_ROOT` 再次运行即可续采。
配额器会读取现有 manifest，只补缺失的 `class × stage × split`，不会覆盖已通过样本。
不要删除 `.collection.lock`；进程退出后内核锁会自动释放。

如需改变输出盘，启动前设置：

```bash
export OUTPUT_ROOT=/大容量磁盘/approach_open_formal_5000_each
```

## 采满后的严格验收

只有启动器报告左右两侧均完成后，才分别运行：

```bash
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/csc/isaacsim/python.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/output/dataset/approach_open_formal_5000_each}"

"$ISAAC_PYTHON" server_tools/audit_completed_dataset.py \
  --camera-root "$OUTPUT_ROOT/right_wrist_d405" \
  --accepted-per-class-stage 250 \
  --report "$OUTPUT_ROOT/right_audit.json"

"$ISAAC_PYTHON" server_tools/audit_completed_dataset.py \
  --camera-root "$OUTPUT_ROOT/left_wrist_d405" \
  --accepted-per-class-stage 250 \
  --report "$OUTPUT_ROOT/left_audit.json"
```

两个命令均须退出码为 0，且报告首项均为 `"status": "PASS"`。审计会检查
每侧 5000 张及精确 split/逐格配额、四件套、尺寸和 dtype、位姿、可见率、深度、
重复样本与 scene group 跨 split。

## 重要约束

- 不要同时对同一个手侧输出目录启动第二个配额器。
- 不要把左右手写入同一个 camera 目录。
- 不要修改 USD、相机外参、左右手参考、可见率阈值或随机化范围。
- 不要使用旧的手背相机、悬空相机、半径扫描或动态 look-at 方案。
- GPU 编号通过 `--gpu-index` 显式传给 Isaac 的 RTX renderer 和 PhysX；
  不应仅依赖 `CUDA_VISIBLE_DEVICES`。
