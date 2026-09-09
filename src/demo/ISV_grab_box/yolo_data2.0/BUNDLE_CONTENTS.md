# 迁移包内容

- `configs/`：资产、相机、沉降、随机化、质量门禁配置。
- `yolo_scene/`：修正后的实验室场景、table_box 和四个工件 USD。
- `references/`：四类工件的右手人工基准及左手确定性镜像参考，共 8 份。
- `yolo_catchdata/`：几何、位姿、质量、随机化、配额和锁模块。
- `scripts/`：计划生成、物理沉降、腕部渲染、配额采集和清洗工具。
- `tests/`：服务器预检使用的回归测试。
- `run_dual_gpu_5000.sh`：GPU 0/1 两手并行正式采集。
- `status_dual_gpu.sh`、`stop_dual_gpu.sh`：状态和安全停止。
- `server_tools/preflight_server.sh`：双卡、依赖、资产、磁盘、测试与 dry-run 预检。
- `server_tools/audit_completed_dataset.py`：每侧采满后的严格数据与配额审计。
- `SERVER_README.md`：上传、预检、启动和续采说明。
- `CODEX_PROMPT.md`：可直接交给服务器 Codex 的任务提示词。
- `MANIFEST.sha256`：除该文件自身及运行输出外的静态文件校验值。

未包含：

- 本机的预采、旧正式采集、rejected 和日志；
- Isaac Sim 安装与 NVIDIA 驱动；
- 与当前侧置腕相机方案无关的上层仓库文件。
