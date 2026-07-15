# 模型3 IMU 独立校准 App 设计

## 目标

为 MaixCAM2 增加一个可由 MaixVision 单独运行、打包和安装的陀螺仪校准 App。它只负责在用户触摸确认后生成 `model3_gimbal` 校准数据，不运行视觉、UART 或姿态解算。模型3主 App 继续只加载校准，不自动覆盖校准数据。

## 现状与入口解释

`模型3/imu_attitude.py` 是库模块，只定义 `ImuAttitude`，没有程序入口或循环，因此在 MaixVision 中“运行当前文件”时会在完成类定义后正常退出。模型3完整程序的入口是 `模型3/main.py`。

独立校准 App 必须有自己的目录和 `main.py`，不能把 `imu_attitude.py` 改成可独立运行脚本，否则会混合“运行时读取”和“写入校准”两个职责。

## 目录与打包

新增同级目录，不改动现有模型3目录结构：

```text
模型3_IMU校准/
├── main.py
└── app.yaml
```

`app.yaml` 使用独立 ASCII ID `model3_imu_calibration`，声明中文名称、版本和作者，并排除 `__pycache__`、`build`、`dist`。MaixVision 打开该目录后以 `main.py` 为项目入口，可直接“运行项目”“打包应用”或“安装应用”。

## 交互流程

App 使用 MaixCAM2 触摸屏，状态固定为以下四种：

1. `READY`：显示静止提示和“开始校准”按钮，不打开 IMU、不写文件。
2. `COUNTDOWN`：触摸按钮后倒计时 3 秒，提示不要移动 MaixCAM2、车体和云台。
3. `CALIBRATING`：创建板载 IMU，显示“校准中，请保持静止”，阻塞采样 10 秒。
4. `SUCCESS` 或 `ERROR`：成功时显示 X/Y/Z 零偏和保存 ID；失败时显示简短错误并提供“点击重试”。成功后不提供再次校准按钮，避免误触重复覆盖。

触摸只在 `READY` 或 `ERROR` 状态、且坐标位于按钮区域时生效。进入倒计时后立即改变状态，持续按压不会重复启动校准。机身默认功能键仍负责退出 App。

## 校准参数与数据共享

校准参数与模型3运行时保持一致：

- 加速度量程：±4 g
- 加速度 ODR：416 Hz
- 陀螺仪量程：±1000 °/s
- 陀螺仪 ODR：416 Hz
- 静止准备时间：3000 ms
- 校准采样时间：10000 ms
- 保存 ID：`model3_gimbal`

调用 `sensor.calib_gyro(10000, save_id="model3_gimbal")` 后，MaixPy 将校准写入设备共享校准目录。模型3 App 中的 `ImuAttitude.start()` 使用相同 ID 调用 `calib_gyro_exists()` 和 `load_calib_gyro()`，因此两个 App 不需要复制文件或共享 Python 模块。

## 错误处理

- App 启动阶段只初始化屏幕和触摸屏；未点击时不访问 IMU。
- IMU 打开或校准异常由一次 `try/except` 捕获，错误界面保留，视觉主程序不受影响。
- 失败不会伪造成功结果；只有 `calib_gyro()` 正常返回才进入 `SUCCESS`。
- 不增加自动重试、后台线程或额外配置文件。

## 验证标准

主机侧验证：

- `app.yaml` ID、入口和排除项正确。
- `main.py` 只在触摸命中后调用校准。
- 校准参数、10 秒时长和 `model3_gimbal` 与模型3运行时一致。
- App 不导入视觉、UART 或 `imu_attitude.py`。
- Python 语法检查和现有测试全部通过。

MaixCAM2 板端验证：

- READY 页面持续显示，不会自动校准或直接退出。
- 按钮外触摸不触发，按钮内触摸只触发一次。
- 倒计时与 10 秒静止校准完成后显示三个零偏。
- 退出校准 App 后运行模型3主 App，状态输出中 `imu_cal=1`。
- 重启设备后再次运行模型3，仍能加载同一份校准。

板端验证完成前，不宣称触摸坐标、屏幕布局或实际校准文件加载已经通过硬件测试。
