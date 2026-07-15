# 模型3 Find Circle 移植设计

## 目标

在不修改已跑通的 `模型3/main.py` 的前提下，将
`2025校赛视觉/find_circle.py` 移植为可独立运行的
`模型3/find_circle.py`，使用模型3目录中的 YOLO11 模型。

## 范围

- 新增 `模型3/find_circle.py`。
- 新增主机侧静态契约测试。
- 保留原来的 OpenCV 二值化、轮廓筛选、透视矫正、圆心和第三圆计算。
- 不修改 `模型3/main.py`，不加入串口、电机、激光或其他控制逻辑。

## 模型与取图适配

- 首选模型路径为 `/root/models/_model25e_maixcam2/best.mud`。
- 首选路径不存在时，使用脚本同目录的 `best.mud`；其引用的两个
  `.axmodel` 文件也已位于同一目录。
- 检测器必须使用 `nn.YOLO11`，不能沿用旧文件的 `nn.YOLOv5`。
- 第一版将 `hires_mode` 设为 `False`，让摄像头使用模型原生输入宽高，
  保持已经由 `模型3/main.py` 验证过的采集和推理尺寸关系。
- `FindRectCircle(disp)`、`get_res()` 和 `run()` 的五项返回值保持不变。

## 验证边界

主机侧验证文件结构、Python 语法、模型加载契约、类接口和返回值。
MaixCAM2 的摄像头、NPU 推理、OpenCV 运行时、画面阈值和实际帧率只能在
开发板上最终验证。
