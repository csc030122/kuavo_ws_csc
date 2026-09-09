你现在位于一台装有两张 NVIDIA GPU 和 Isaac Sim 6.0 的 Linux 服务器上。用户已经上传
`yolo_catchdata_server_dual_gpu_5000` 目录。请直接完成双 GPU 正式采集的部署、启动、
监控与最终验收，不要重新设计采集方案。

目标：

1. 右腕侧置 D405 使用物理 GPU 0，独立采集 5000 张。
2. 左腕侧置 D405 使用物理 GPU 1，独立采集 5000 张。
3. 每只手均为 4 类（left/right/hose/outhandle）× 5 层
   （grasp/pre_grasp/near/approach/far）× 每格 250 张。
4. 每只手最终 train/val/test 必须为 3500/750/750；同一稳定沉降 scene group
   不得跨 split。
5. 两侧同时运行，但计划、manifest、输出目录、稳定姿态命名空间和文件锁必须独立。

执行要求：

- 先阅读 `SERVER_README.md`，确认当前目录就是上传包根目录。
- 用 `nvidia-smi` 确认至少两张 GPU；确认两卡当前没有其它高显存任务。
- 若 Isaac Sim 不在 `/home/csc/isaacsim/python.sh`，找到实际 `python.sh` 并设置
  `ISAAC_PYTHON`，不要修改代码里的场景、相机或采集参数来绕过环境问题。
- 运行 `./server_tools/preflight_server.sh`。只有看到 `PREFLIGHT=PASS` 才能启动。
- 使用 `run_dual_gpu_5000.sh`，默认 RIGHT_GPU=0、LEFT_GPU=1。后台启动前确认没有旧
  PID 和同目录活跃锁，避免重复运行。
- 启动后用 `nvidia-smi` 和两个分支日志确认右手确实占用 GPU 0、左手确实占用 GPU 1，
  且两边都至少产生一条 `PASS stable=pass render=pass`。
- 采集中可用 `./status_dual_gpu.sh` 查看进度。状态未变化并不意味着失败，先检查当前
  Isaac 子进程和日志时间。不要因普通 quality rejection 停止任务；配额器会自动补采。
- 如进程异常退出，先诊断原因；环境恢复后用完全相同的 OUTPUT_ROOT 再次启动，利用
  manifest 断点续采。不要新建另一个目录来掩盖可恢复的中断。

禁止事项：

- 不要修改 `configs/collection.yaml` 的阈值、距离层或随机化。
- 不要修改场景 USD、四个工件 USD、相机固定外参或 `references/` 中的标定。
- 不要恢复旧的手背上方相机、悬空视角、theta/phi 半径扫描或动态 look-at。
- 不要合并左右手 manifest，不要让两个进程写同一个 camera 根目录。
- 不要把 rejected 文件移动进 train/val/test。

完成后进行严格验收并向用户报告：

- 每侧 accepted 总数、四类×五层的逐格数量、train/val/test 数量；
- RGB/depth/mask/pose 四件套完整性；
- RGB 为 1280×720 三通道 PNG，Mask 仅含 0/255，Depth 为 720×1280 float32 NPY；
- pose.json 含 R_vec(a,b,c)、t_vec(x,y,z)，旋转矩阵正交且 det≈+1，数值有限；
- Mask 像素数、Depth 有效率、真实场景有效可见率均满足当前配置；
- 无重复轨迹、无重复 RGB、无 scene group 跨 split；
- rejected 和 stable failure 按原因汇总；
- 两个分支最终都显示严格配额达到。如果任一项不满足，不要宣称完成。

启动器完成后，必须从上传包根目录分别运行以下机器审计；保留两个 JSON 报告：

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

两个报告都必须是 `"status": "PASS"` 且命令退出码为 0；否则继续诊断或补采，
不要仅凭文件数量宣称完成。

先执行环境与文件预检，然后启动，不要只把命令发给用户。
