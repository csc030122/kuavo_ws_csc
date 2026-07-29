# S63 实验室场景迁移到 Isaac Sim 6

这一版用于快速确认迁移效果：直接让 Isaac Sim 6 的 MJCF Importer 读取
`lab_sence_static_with_box.xml`，并生成 USD。它会带入当前 MJCF 中的 S63
轮臂机器人、实验室、货架、托盘、45 个自由箱体、AprilTag 贴图和摄像机定义。

这不是最终的高保真标定版。MuJoCo 与 PhysX 的接触、摩擦、阻尼和执行器模型并
不完全相同；第一轮的目标是先检查比例、坐标、关节、碰撞和贴图是否完整。

## 1. 确认代码目录已挂载到容器

以下命令在服务器宿主机执行：

```bash
docker inspect isaac-sim \
  --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
```

输出中应包含本项目目录，并且挂载模式是可写。记下箭头右侧的容器内路径。

## 2. 导入 MJCF

进入容器：

```bash
docker exec -it isaac-sim bash
```

在容器中进入本目录并执行：

```bash
cd <容器内项目路径>/src/demo/ISV_grab_box
bash isaac_sim/import_mjcf.sh
```

脚本默认使用 `/isaac-sim/python.sh`，所以不要在宿主机直接运行
`"$ISAAC_SIM_ROOT/python.sh"`。导入输出位于：

```text
isaac_sim/generated/mjcf_baseline_年月日_时分秒/
```

每次运行使用新目录，不会覆盖上一次结果。脚本还会先检查所有 XML、STL 和
AprilTag PNG 是否存在。需要额外传入官方导入器参数时，可以直接追加，例如：

```bash
bash isaac_sim/import_mjcf.sh --merge-mesh
```

第一轮不建议使用 `--collision-from-visuals`，因为当前 MJCF 已有显式碰撞几何。

## 3. 查看效果

最稳妥的方式是在当前已经运行的 Isaac Sim GUI 中选择 **File > Open**，打开
刚生成目录中的主 USD。

如果容器允许启动第二个 Kit/GUI 实例，也可以运行：

```bash
bash isaac_sim/preview_usd.sh
```

无窗口检查：

```bash
bash isaac_sim/preview_usd.sh --headless --seconds 10
```

指定某个 USD：

```bash
bash isaac_sim/preview_usd.sh \
  --usd /容器内绝对路径/lab_sence_static_with_box.usd
```

## 4. 第一轮验收

依次检查：

1. S63 出现在 `(1.5, 0, 0)` 附近，朝向和 MuJoCo 一致。
2. 4 个轮组、14 个手臂关节及固定 D-hand 外观都存在。
3. 地面 AprilTag 09–18、货架 AprilTag 01–04 可见且方向正确。
4. 货架上 9 个箱体及托盘上 36 个箱体齐全；目标箱
   `lab_target_box` 在 `(2.454, 0, 0.780)` 附近。
5. 点击 Play 后机器人或箱体没有爆飞、穿透、持续抖动。
6. 基座、头部和左右手腕摄像机是否保留；导入器未保留时将在第二阶段按传感器
   标定参数重建。

如发生爆飞，先暂停仿真并保存 Console 日志与问题画面，不要先随意调质量或
摩擦参数；这些信息用于定位 PhysX 接触、惯量或关节驱动的差异。

## 5. 后续高保真迁移需要的资料

- S63 与实际装配一致的 URDF/Xacro、mesh、关节限位、减速比和电机参数。
- 底盘轮组结构、轮径、轮距、转向零位及控制方式。
- 各相机的型号、内参、畸变、分辨率、帧率及相对机器人坐标系外参。
- 实验室和货架的实测尺寸、照片/视频、材质与照明条件。
- 箱体实测尺寸、质量、质心，以及接触面的摩擦系数或可用于辨识的实验数据。
- MuJoCo 中一次正常运行的录屏、初始状态和控制输入日志。

