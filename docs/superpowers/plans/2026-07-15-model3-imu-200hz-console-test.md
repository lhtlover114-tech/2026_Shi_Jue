# 模型3 IMU 200 Hz 控制台测试 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个不启动视觉和 UART 的独立入口，以 5 ms 绝对截止时间调度现有 `ImuAttitude`，逐条打印 IMU 姿态并汇报真实频率和超期数。

**Architecture:** `模型3/imu_attitude_test.py` 只负责测试调度、输出和统计，所有硬件初始化、校准加载和姿态解算继续由 `ImuAttitude` 完成。MaixPy 硬件模块延迟到 `main()` 内导入，使主机测试可以用假时间、假退出源和假姿态源验证 200 Hz 行为。

**Tech Stack:** MaixPy Python、`maix.app`、`maix.time`、Python `unittest`

## Global Constraints

- 测试入口不得调用 `calib_gyro()` 或 `save_calib_gyro()`，只允许 `ImuAttitude.start()` 加载 `model3_gimbal`。
- 周期固定为 `5000 us`，使用绝对截止时间累加，不能用每轮“当前时间加 5000 us”。
- 每个成功样本打印一行；每 `1000000 us` 汇报成功采样频率、数据打印频率、错误数和累计跳过时隙。
- 不导入或启动 `find_circle`、`maixcam_link`、UART 和模型推理。
- 不修改 `imu_attitude.py`、`main.py`、`find_circle.py` 或 `maixcam_link.py` 的接口和行为。

---

### Task 1: 用失败测试定义控制台测试入口契约

**Files:**
- Create: `tests/test_model3_imu_attitude_console.py`
- Test: `tests/test_model3_imu_attitude_console.py`

**Interfaces:**
- Consumes: `ImuAttitude.start() -> bool` 和 `ImuAttitude.sample() -> tuple[int, int, int, int, int]`
- Produces: 对待新增 `format_sample(sequence, fields) -> str` 与 `run_console_test(attitude, app_module, time_module, print_fn=print, period_us=5000, report_interval_us=1000000) -> tuple[int, int, int, int]` 的可执行契约

- [ ] **Step 1: 写入失败测试**

```python
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL3_DIR = ROOT / "模型3"
CONSOLE_PATH = MODEL3_DIR / "imu_attitude_test.py"


def load_console_module():
    sys.path.insert(0, str(MODEL3_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "model3_imu_attitude_console", CONSOLE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class FakeAttitude:
    def __init__(self, samples=(), start_result=True):
        self.samples = list(samples)
        self.start_result = start_result
        self.sample_calls = 0

    def start(self):
        return self.start_result

    def sample(self):
        self.sample_calls += 1
        if self.samples:
            value = self.samples.pop(0)
            if isinstance(value, Exception):
                raise value
            return value
        return (12, -4, 1235, -210, 0x06)


class FakeApp:
    def __init__(self, iterations):
        self.iterations = iterations
        self.calls = 0

    def need_exit(self):
        self.calls += 1
        return self.calls > self.iterations


class FakeTime:
    def __init__(self):
        self.now_us = 0
        self.sleeps = []

    def ticks_us(self):
        return self.now_us

    def sleep_us(self, duration_us):
        self.sleeps.append(duration_us)
        self.now_us += duration_us


class Model3ImuAttitudeConsoleTests(unittest.TestCase):
    def test_formats_fixed_point_fields_as_engineering_units(self):
        module = load_console_module()

        self.assertEqual(
            module.format_sample(123, (12, -4, 1235, -210, 0x06)),
            "imu[00123] yr=+1.2 pr=-0.4 ya=+12.35 pa=-2.10 flags=0x06",
        )

    def test_missing_calibration_exits_without_sampling(self):
        module = load_console_module()
        attitude = FakeAttitude(start_result=False)
        output = []

        stats = module.run_console_test(
            attitude,
            FakeApp(3),
            FakeTime(),
            print_fn=output.append,
        )

        self.assertEqual(stats, (0, 0, 0, 0))
        self.assertEqual(attitude.sample_calls, 0)
        self.assertTrue(any("model3_gimbal" in line for line in output))

    def test_uses_absolute_five_millisecond_deadlines(self):
        module = load_console_module()
        attitude = FakeAttitude()
        fake_time = FakeTime()
        output = []

        stats = module.run_console_test(
            attitude,
            FakeApp(4),
            fake_time,
            print_fn=output.append,
        )

        self.assertEqual(stats, (4, 4, 0, 0))
        self.assertEqual(fake_time.sleeps, [5000, 5000, 5000])
        self.assertEqual(
            [line for line in output if line.startswith("imu[")],
            [
                "imu[00000] yr=+1.2 pr=-0.4 ya=+12.35 pa=-2.10 flags=0x06",
                "imu[00001] yr=+1.2 pr=-0.4 ya=+12.35 pa=-2.10 flags=0x06",
                "imu[00002] yr=+1.2 pr=-0.4 ya=+12.35 pa=-2.10 flags=0x06",
                "imu[00003] yr=+1.2 pr=-0.4 ya=+12.35 pa=-2.10 flags=0x06",
            ],
        )

    def test_reports_200_hz_before_sampling_next_window(self):
        module = load_console_module()
        output = []

        stats = module.run_console_test(
            FakeAttitude(),
            FakeApp(201),
            FakeTime(),
            print_fn=output.append,
        )

        self.assertEqual(stats, (201, 201, 0, 0))
        self.assertIn(
            "[rate] sample=200.0Hz print=200.0Hz errors=0 skipped=0",
            output,
        )

    def test_slow_console_counts_expired_slots(self):
        module = load_console_module()
        fake_time = FakeTime()
        output = []

        def slow_print(line):
            output.append(line)
            fake_time.now_us += 12000

        stats = module.run_console_test(
            FakeAttitude(),
            FakeApp(1),
            fake_time,
            print_fn=slow_print,
        )

        self.assertEqual(stats, (1, 1, 0, 2))

    def test_sample_exception_is_counted_and_next_slot_continues(self):
        module = load_console_module()
        attitude = FakeAttitude(
            samples=[RuntimeError("read failed"), (12, -4, 1235, -210, 0x06)]
        )
        output = []

        stats = module.run_console_test(
            attitude,
            FakeApp(2),
            FakeTime(),
            print_fn=output.append,
        )

        self.assertEqual(stats, (1, 1, 1, 0))
        self.assertTrue(any("read failed" in line for line in output))
        self.assertEqual(
            len([line for line in output if line.startswith("imu[")]),
            1,
        )

    def test_entry_does_not_own_calibration_uart_or_vision(self):
        text = CONSOLE_PATH.read_text(encoding="utf-8")

        self.assertNotIn("calib_gyro(", text)
        self.assertNotIn("save_calib_gyro(", text)
        self.assertNotIn("find_circle", text)
        self.assertNotIn("maixcam_link", text)
        self.assertNotIn("uart", text.lower())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认因入口缺失而失败**

Run: `python -m unittest tests.test_model3_imu_attitude_console -v`

Expected: FAIL/ERROR，明确指出 `模型3/imu_attitude_test.py` 不存在，证明测试覆盖的是尚未实现的新入口。

- [ ] **Step 3: 暂存并中文提交失败测试**

```bash
git add tests/test_model3_imu_attitude_console.py
git commit -m "测试：定义模型3 IMU 200Hz控制台契约"
```

### Task 2: 实现独立 200 Hz IMU 控制台入口

**Files:**
- Create: `模型3/imu_attitude_test.py`
- Test: `tests/test_model3_imu_attitude_console.py`

**Interfaces:**
- Consumes: `ImuAttitude.start()`、`ImuAttitude.sample()`、`app.need_exit()`、`time.ticks_us()`、`time.sleep_us()`
- Produces: `format_sample(sequence, fields) -> str`；`run_console_test(...) -> (samples, prints, errors, skipped)`；MaixVision 可直接运行的 `main()`

- [ ] **Step 1: 写入最小实现**

```python
# -*- coding: utf-8 -*-
"""独立验证模型3板载 IMU 能否维持 200 Hz 姿态采样与控制台输出。"""

from imu_attitude import ImuAttitude


PERIOD_US = 5000
REPORT_INTERVAL_US = 1000000
CALIBRATION_SAVE_ID = "model3_gimbal"


def format_sample(sequence, fields):
    yaw_rate_x10, pitch_rate_x10, yaw_x100, pitch_x100, flags = fields
    return (
        "imu[{:05d}] yr={:+.1f} pr={:+.1f} "
        "ya={:+.2f} pa={:+.2f} flags=0x{:02X}"
    ).format(
        int(sequence),
        yaw_rate_x10 / 10.0,
        pitch_rate_x10 / 10.0,
        yaw_x100 / 100.0,
        pitch_x100 / 100.0,
        int(flags) & 0xFF,
    )


def run_console_test(
    attitude,
    app_module,
    time_module,
    print_fn=print,
    period_us=PERIOD_US,
    report_interval_us=REPORT_INTERVAL_US,
):
    try:
        if not attitude.start():
            print_fn(
                "[imu-test] calibration model3_gimbal not found; "
                "run the calibration app first"
            )
            return (0, 0, 0, 0)
    except Exception as exc:
        print_fn("[imu-test] initialization failed: {}".format(exc))
        return (0, 0, 1, 0)

    print_fn("[imu-test] started: period=5000us, console=every sample")
    sample_count = 0
    print_count = 0
    error_count = 0
    skipped_count = 0
    sequence = 0
    next_deadline_us = time_module.ticks_us()
    report_start_us = next_deadline_us
    report_sample_count = 0
    report_print_count = 0

    while not app_module.need_exit():
        now_us = time_module.ticks_us()
        if now_us < next_deadline_us:
            time_module.sleep_us(next_deadline_us - now_us)
        now_us = time_module.ticks_us()

        elapsed_us = now_us - report_start_us
        if elapsed_us >= report_interval_us:
            elapsed_s = elapsed_us / 1000000.0
            print_fn(
                "[rate] sample={:.1f}Hz print={:.1f}Hz "
                "errors={} skipped={}".format(
                    (sample_count - report_sample_count) / elapsed_s,
                    (print_count - report_print_count) / elapsed_s,
                    error_count,
                    skipped_count,
                )
            )
            report_start_us = now_us
            report_sample_count = sample_count
            report_print_count = print_count

        try:
            fields = attitude.sample()
            sample_count += 1
            print_fn(format_sample(sequence, fields))
            print_count += 1
        except Exception as exc:
            error_count += 1
            if error_count == 1 or error_count % 200 == 0:
                print_fn("[imu-test] sample failed: {}".format(exc))

        sequence += 1
        next_deadline_us += period_us
        now_us = time_module.ticks_us()
        while next_deadline_us <= now_us:
            next_deadline_us += period_us
            skipped_count += 1

    return (sample_count, print_count, error_count, skipped_count)


def main():
    from maix import app, time

    run_console_test(ImuAttitude(), app, time)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行专用测试并确认通过**

Run: `python -m unittest tests.test_model3_imu_attitude_console -v`

Expected: 7 tests PASS。

- [ ] **Step 3: 运行现有模型3测试并确认没有回归**

Run: `python -m unittest tests.test_model3_imu_attitude tests.test_model3_maixcam_link tests.test_model3_imu_calibration tests.test_model3_imu_attitude_console -v`

Expected: 所有测试 PASS；现有 `imu_attitude.py` 和最终 200 Hz 合帧链路契约不变。

- [ ] **Step 4: 检查语法、空白和工作区范围**

Run: `python -m py_compile "模型3/imu_attitude_test.py" "tests/test_model3_imu_attitude_console.py"`

Expected: exit code 0。

Run: `git diff --check`

Expected: exit code 0。

Run: `git status --short`

Expected: 只有 `模型3/imu_attitude_test.py` 为未暂存实现文件；失败测试已经由 Task 1 单独提交。

- [ ] **Step 5: 暂存并中文提交实现**

```bash
git add "模型3/imu_attitude_test.py"
git commit -m "功能：增加模型3 IMU 200Hz控制台测试"
```

### Task 3: 完成最终验收审计

**Files:**
- Verify: `模型3/imu_attitude_test.py`
- Verify: `模型3/imu_attitude.py`
- Verify: `模型3/main.py`
- Verify: `模型3/maixcam_link.py`
- Test: `tests/test_model3_imu_attitude_console.py`

**Interfaces:**
- Consumes: Task 1 和 Task 2 的已提交测试与实现
- Produces: 可供用户复制到 MaixVision 并上板运行的明确验证结论和剩余硬件验证边界

- [ ] **Step 1: 运行仓库完整测试**

Run: `python -m unittest discover -s tests -v`

Expected: 全部测试 PASS。

- [ ] **Step 2: 核对范围约束**

Run: `rg -n "calib_gyro\(|save_calib_gyro\(|find_circle|maixcam_link|uart" "模型3/imu_attitude_test.py"`

Expected: 无匹配，exit code 1；测试入口不拥有校准、视觉或 UART。

Run: `git status --short --branch`

Expected: 工作区干净。

- [ ] **Step 3: 上板交付说明**

在 MaixVision 中单独运行 `模型3/imu_attitude_test.py`。验证控制台每 5 ms 输出一条 `imu[...]`，每秒输出一条 `[rate]`；记录 `sample`、`print`、`errors` 和 `skipped`。主机测试只能证明调度逻辑和模块边界，MaixCAM2 是否能承受 200 行/秒必须以上板日志为最终证据。
