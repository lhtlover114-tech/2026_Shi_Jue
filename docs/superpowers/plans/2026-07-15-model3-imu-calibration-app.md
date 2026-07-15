# 模型3 IMU 独立校准 App 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个可由 MaixVision 独立运行、打包和安装的触摸屏 IMU 校准 App，仅在用户点击确认后保存 `model3_gimbal` 陀螺仪零偏。

**Architecture:** 新目录 `模型3_IMU校准` 独立于视觉模型3 App，包含自己的 `main.py` 和 `app.yaml`。`main.py` 使用一个小型 READY/COUNTDOWN/CALIBRATING/SUCCESS/ERROR 状态机；Maix 模块在入口中注入到 `_run_app()`，便于主机测试确认未触摸时绝不打开 IMU。

**Tech Stack:** MaixPy、`maix.touchscreen`、`maix.display`、`maix.image`、`maix.ext_dev.imu`、Python `unittest`。

## Global Constraints

- 不修改 `模型3`、`Maixcam2` 或 `2025校赛视觉` 的现有运行代码和目录结构。
- App ID 固定为 `model3_imu_calibration`，入口固定为 `模型3_IMU校准/main.py`。
- READY 状态不得创建 IMU、采样或写入校准文件。
- 只有触摸按下沿命中按钮区域后才允许校准，持续按压不得重复触发。
- 使用 `save_id="model3_gimbal"`、10000 ms 校准、3000 ms 倒计时、±4 g、±1000 °/s、加速度/陀螺仪 ODR 416 Hz。
- 成功后不提供再次校准按钮；失败后必须先松手，再次点击才可重试。
- 不导入视觉、UART、模型或 `imu_attitude.py`。
- 暂存说明和 Git 提交信息使用中文，只暂存本计划涉及的文件。

---

### Task 1: 建立独立 App 的可测试行为契约

**Files:**
- Create: `tests/test_model3_imu_calibration_app.py`
- Create: `模型3_IMU校准/main.py`
- Create: `模型3_IMU校准/app.yaml`

**Interfaces:**
- Consumes: MaixPy `app.need_exit()`、`touchscreen.TouchScreen.read()`、`display.Display`、`image.Image`、`imu.IMU`。
- Produces: `main() -> int`、`_run_app(app, display, image, time, touchscreen, imu) -> int`、`_point_in_rect(x, y, rect) -> bool`。

- [ ] **Step 1: 写入失败的 App 契约测试**

创建 `tests/test_model3_imu_calibration_app.py`，测试必须覆盖：

```python
import importlib.util
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "模型3_IMU校准"
MAIN_PATH = APP_DIR / "main.py"
MANIFEST_PATH = APP_DIR / "app.yaml"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "model3_imu_calibration_app", MAIN_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCanvas:
    def draw_rect(self, *args, **kwargs):
        pass

    def draw_string(self, *args, **kwargs):
        pass


class FakeImageModule:
    COLOR_BLACK = 0
    COLOR_WHITE = 1
    COLOR_GREEN = 2
    COLOR_RED = 3
    COLOR_BLUE = 4

    @staticmethod
    def Image(width, height):
        return FakeCanvas()


class FakeDisplay:
    def __init__(self):
        self.frames = []

    def width(self):
        return 640

    def height(self):
        return 480

    def show(self, frame):
        self.frames.append(frame)


class FakeDisplayModule:
    def __init__(self):
        self.instance = FakeDisplay()

    def Display(self):
        return self.instance


class FakeTouch:
    def __init__(self, events):
        self.events = list(events)

    def read(self):
        if self.events:
            return self.events.pop(0)
        return (0, 0, False)


class FakeTouchscreenModule:
    def __init__(self, events):
        self.instance = FakeTouch(events)

    def TouchScreen(self):
        return self.instance


class FakeApp:
    def __init__(self, exit_after):
        self.exit_after = exit_after
        self.checks = 0

    def need_exit(self):
        self.checks += 1
        return self.checks > self.exit_after


class FakeTime:
    def __init__(self):
        self.sleeps = []

    def sleep_ms(self, value):
        self.sleeps.append(value)


def make_imu_module():
    sensor = mock.Mock()
    sensor.calib_gyro.return_value = types.SimpleNamespace(
        x=0.1, y=-0.2, z=0.3
    )
    module = types.SimpleNamespace(
        Mode=types.SimpleNamespace(DUAL="dual"),
        AccScale=types.SimpleNamespace(ACC_SCALE_4G="4g"),
        AccOdr=types.SimpleNamespace(ACC_ODR_416="acc416"),
        GyroScale=types.SimpleNamespace(GYRO_SCALE_1000DPS="1000dps"),
        GyroOdr=types.SimpleNamespace(GYRO_ODR_416="gyro416"),
        IMU=mock.Mock(return_value=sensor),
    )
    return module, sensor


class Model3ImuCalibrationAppTests(unittest.TestCase):
    def test_manifest_and_source_contract(self):
        self.assertTrue(MAIN_PATH.exists())
        self.assertTrue(MANIFEST_PATH.exists())
        manifest = MANIFEST_PATH.read_text(encoding="utf-8")
        source = MAIN_PATH.read_text(encoding="utf-8")
        self.assertIn("id: model3_imu_calibration", manifest)
        self.assertIn("name[zh]: 模型3 IMU 校准", manifest)
        self.assertIn("- __pycache__", manifest)
        self.assertNotIn("find_circle", source)
        self.assertNotIn("maixcam_link", source)
        self.assertNotIn("imu_attitude", source)

    def test_no_touch_exits_without_opening_imu(self):
        module = load_module()
        imu_module, sensor = make_imu_module()
        result = module._run_app(
            FakeApp(exit_after=1),
            FakeDisplayModule(),
            FakeImageModule(),
            FakeTime(),
            FakeTouchscreenModule([(0, 0, False)]),
            imu_module,
        )
        self.assertEqual(result, 0)
        imu_module.IMU.assert_not_called()
        sensor.calib_gyro.assert_not_called()

    def test_touch_inside_button_runs_fixed_calibration_once(self):
        module = load_module()
        imu_module, sensor = make_imu_module()
        fake_time = FakeTime()
        result = module._run_app(
            FakeApp(exit_after=4),
            FakeDisplayModule(),
            FakeImageModule(),
            fake_time,
            FakeTouchscreenModule([(320, 340, True)]),
            imu_module,
        )
        self.assertEqual(result, 0)
        imu_module.IMU.assert_called_once_with(
            "default",
            mode="dual",
            acc_scale="4g",
            acc_odr="acc416",
            gyro_scale="1000dps",
            gyro_odr="gyro416",
        )
        sensor.calib_gyro.assert_called_once_with(
            10000, save_id="model3_gimbal"
        )
        self.assertEqual(fake_time.sleeps[:3], [1000, 1000, 1000])

    def test_failure_requires_release_before_touch_retry(self):
        module = load_module()
        imu_module, sensor = make_imu_module()
        sensor.calib_gyro.side_effect = [
            RuntimeError("imu failed"),
            types.SimpleNamespace(x=0.1, y=-0.2, z=0.3),
        ]
        result = module._run_app(
            FakeApp(exit_after=10),
            FakeDisplayModule(),
            FakeImageModule(),
            FakeTime(),
            FakeTouchscreenModule(
                [
                    (320, 340, True),
                    (320, 340, True),
                    (320, 340, False),
                    (320, 340, True),
                ]
            ),
            imu_module,
        )
        self.assertEqual(result, 0)
        self.assertEqual(imu_module.IMU.call_count, 2)
        self.assertEqual(sensor.calib_gyro.call_count, 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```powershell
& 'C:\Users\34697\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B -m unittest tests.test_model3_imu_calibration_app -v
```

Expected: FAIL，原因是 `模型3_IMU校准/main.py` 和 `app.yaml` 尚不存在。

- [ ] **Step 3: 写入最小 App 清单**

创建 `模型3_IMU校准/app.yaml`：

```yaml
id: model3_imu_calibration
name: Model3 IMU Calibration
name[zh]: 模型3 IMU 校准
version: 1.0.0
author: LLL
desc: Calibrate and save the Model3 MaixCAM2 gyro bias
exclude:
  - __pycache__
  - build
  - dist
  - .gitignore
```

- [ ] **Step 4: 实现触摸状态机和校准动作**

创建 `模型3_IMU校准/main.py`，保留以下固定接口和控制流：

```python
# -*- coding: utf-8 -*-
"""Standalone touchscreen gyro calibration app for Model 3."""


CALIBRATION_TIME_MS = 10000
CALIBRATION_SAVE_ID = "model3_gimbal"
COUNTDOWN_SECONDS = 3

STATE_READY = "ready"
STATE_ERROR = "error"
STATE_SUCCESS = "success"


def _button_rect(width, height):
    button_width = width // 2
    button_height = 80
    return (
        (width - button_width) // 2,
        height - button_height - 60,
        button_width,
        button_height,
    )


def _point_in_rect(x, y, rect):
    left, top, width, height = rect
    return left <= x < left + width and top <= y < top + height


def _create_sensor(imu):
    return imu.IMU(
        "default",
        mode=imu.Mode.DUAL,
        acc_scale=imu.AccScale.ACC_SCALE_4G,
        acc_odr=imu.AccOdr.ACC_ODR_416,
        gyro_scale=imu.GyroScale.GYRO_SCALE_1000DPS,
        gyro_odr=imu.GyroOdr.GYRO_ODR_416,
    )


def _draw(disp, image, title, lines, button_label=None, success=False):
    width = disp.width()
    height = disp.height()
    canvas = image.Image(width, height)
    canvas.draw_rect(0, 0, width, height, image.COLOR_BLACK, thickness=-1)
    title_color = image.COLOR_GREEN if success else image.COLOR_WHITE
    canvas.draw_string(40, 35, title, title_color, scale=2)
    y = 110
    for line in lines:
        canvas.draw_string(40, y, line, image.COLOR_WHITE, scale=1.5)
        y += 45
    if button_label:
        rect = _button_rect(width, height)
        canvas.draw_rect(*rect, image.COLOR_BLUE, thickness=-1)
        canvas.draw_string(
            rect[0] + 30,
            rect[1] + 24,
            button_label,
            image.COLOR_WHITE,
            scale=1.5,
        )
    disp.show(canvas)


def _calibrate(imu):
    sensor = _create_sensor(imu)
    return sensor.calib_gyro(
        CALIBRATION_TIME_MS,
        save_id=CALIBRATION_SAVE_ID,
    )


def _run_app(app, display, image, time, touchscreen, imu):
    disp = display.Display()
    touch = touchscreen.TouchScreen()
    button = _button_rect(disp.width(), disp.height())
    state = STATE_READY
    was_pressed = False
    _draw(
        disp,
        image,
        "Model3 IMU Calibration",
        ["Keep camera, chassis and gimbal STILL."],
        "START CALIBRATION",
    )

    while not app.need_exit():
        x, y, pressed = touch.read()
        pressed = bool(pressed)
        trigger = (
            state in (STATE_READY, STATE_ERROR)
            and pressed
            and not was_pressed
            and _point_in_rect(x, y, button)
        )
        was_pressed = pressed

        if trigger:
            for remaining in range(COUNTDOWN_SECONDS, 0, -1):
                _draw(
                    disp,
                    image,
                    "Keep Still",
                    ["Calibration starts in {}...".format(remaining)],
                )
                if app.need_exit():
                    return 0
                time.sleep_ms(1000)

            _draw(
                disp,
                image,
                "Calibrating",
                ["Do not move for 10 seconds."],
            )
            try:
                bias = _calibrate(imu)
            except Exception as exc:
                state = STATE_ERROR
                _draw(
                    disp,
                    image,
                    "Calibration Failed",
                    [str(exc)[:64]],
                    "TOUCH TO RETRY",
                )
            else:
                state = STATE_SUCCESS
                _draw(
                    disp,
                    image,
                    "Calibration Success",
                    [
                        "X: {:.6f}".format(bias.x),
                        "Y: {:.6f}".format(bias.y),
                        "Z: {:.6f}".format(bias.z),
                        "Saved as: {}".format(CALIBRATION_SAVE_ID),
                    ],
                    success=True,
                )

        time.sleep_ms(20)
    return 0


def main():
    from maix import app, display, image, time, touchscreen
    from maix.ext_dev import imu

    return _run_app(app, display, image, time, touchscreen, imu)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: 运行聚焦测试并确认 GREEN**

Run:

```powershell
& 'C:\Users\34697\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B -m unittest tests.test_model3_imu_calibration_app -v
```

Expected: 4 tests PASS。

- [ ] **Step 6: 中文暂存并提交 App 与测试**

```powershell
git add -- '模型3_IMU校准/main.py' '模型3_IMU校准/app.yaml' 'tests/test_model3_imu_calibration_app.py'
git commit -m '功能：增加模型3独立IMU校准应用'
```

---

### Task 2: 回归验证和打包交付审计

**Files:**
- Verify: `模型3_IMU校准/main.py`
- Verify: `模型3_IMU校准/app.yaml`
- Verify: `模型3/imu_attitude.py`
- Verify: `tests/test_model3_imu_calibration_app.py`

**Interfaces:**
- Consumes: Task 1 的独立 App 和模型3运行时保存 ID。
- Produces: 主机验证证据和明确的 MaixCAM2 板端测试边界。

- [ ] **Step 1: 运行全部主机测试**

Run:

```powershell
& 'C:\Users\34697\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B -m unittest discover -s tests -v
```

Expected: 原 17 项测试和 4 项新 App 测试全部 PASS。

- [ ] **Step 2: 检查语法与固定契约**

Run:

```powershell
& 'C:\Users\34697\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B -c "import ast,pathlib; files=[pathlib.Path('模型3_IMU校准/main.py'), pathlib.Path('模型3/imu_attitude.py')]; [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print('AST syntax OK:', len(files))"
rg -n "model3_imu_calibration|model3_gimbal|10000|ACC_SCALE_4G|ACC_ODR_416|GYRO_SCALE_1000DPS|GYRO_ODR_416" '模型3_IMU校准' '模型3/imu_attitude.py'
git diff --check
```

Expected: AST 检查 2 个文件通过；App ID、保存 ID 和全部固定参数均能检索到；`git diff --check` 返回 0。

- [ ] **Step 3: 审计提交与用户改动边界**

Run:

```powershell
git status --short
git log -3 --oneline
```

Expected: 新 App 文件已提交；`模型3/main.py` 和 `模型3/maixcam_link.py` 原有中文注释改动仍未被暂存或覆盖；没有测试生成的 `pyc` 残留。

- [ ] **Step 4: 交付板端验证步骤**

在最终报告中明确：

1. MaixVision 打开 `模型3_IMU校准` 目录并使用“运行项目”，不能只运行库文件。
2. READY 页面静置 10 秒不得自动进入校准。
3. 点击按钮后完成 3 秒倒计时和 10 秒校准，记录 X/Y/Z 零偏。
4. 退出后运行模型3主 App，确认状态行 `imu_cal=1`。
5. 重启设备后再次运行模型3，确认仍为 `imu_cal=1`。

不得把主机测试表述成触摸坐标、板载 IMU 或持久化文件已经通过硬件验证。
