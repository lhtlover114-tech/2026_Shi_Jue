# MaixCAM2 `find_circle.py` 移植设计

## 目标

将 `2025校赛视觉/find_circle.py` 移植为可在 MaixCAM2 上独立运行的
`Maixcam2/find_circle.py`，使用已经在 `Maixcam2/main.py` 中跑通的
MaixCAM2 模型。

## 范围

- 新增 `Maixcam2/find_circle.py`。
- 不修改 `Maixcam2/main.py`，本轮不做应用入口集成。
- 不加入通信上报、电机、激光或其他控制逻辑。
- 不重构原有 OpenCV 视觉算法。

## 接口与行为

- 保留 `FindRectCircle(disp)` 类接口。
- 保留 `get_res()`。
- 保留 `run()` 的五项返回值：
  `[last_center, center_pos, err_center, circle3_points, updated]`。
- 保留 448 x 448 高分辨率取图、YOLOv5 矩形检测、透视矫正、黑框轮廓
  筛选、圆心与第三圆计算、调试绘制和显示逻辑。
- 保留文件末尾的独立运行入口，循环调用 `run()` 并打印 `err_center`。

## MaixCAM2 适配

- 首选模型路径：`/root/models/_model25e_maixcam2/model_246619.mud`。
- 首选路径不存在时，尝试脚本同目录下的 `model_246619.mud`。
- 模型加载继续使用 `nn.YOLOv5`；高分辨率取图保持 448 x 448，输入格式由
  模型元数据提供。
- 不再引用 MaixCAM1 的 `model_3356.mud`。

## 验证

- 测试先验证目标文件尚不存在或尚未满足移植契约。
- 移植后运行主机侧语法编译。
- 静态检查类接口、五项返回值、独立入口、新模型路径以及旧模型引用已移除。
- 检查 Git 差异，确认 `Maixcam2/main.py` 未被修改。
- 主机没有 MaixCAM2 的 `maix` 运行时和 NPU，因此摄像头、模型推理、显示与
  帧率只能在开发板上最终确认。
