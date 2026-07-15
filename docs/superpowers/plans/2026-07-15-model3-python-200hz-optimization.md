# 模型3纯 Python 200 Hz 优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在不改变模型3现有 28 字节串口协议、不新增进程/线程和不改项目结构的前提下，把 MaixCAM2 端 UART4 提升到 460800，并通过 CRC 查表、1 ms Python 线程切换间隔、视觉循环主动让出执行权和调试绘制减负，争取让 IMU 采样与整帧发送稳定在 200 Hz，同时保留 30 FPS 以上视觉和候选矩形显示。

**Architecture:** 保留 `模型3/main.py` 的视觉生产者和 `模型3/maixcam_link.py` 的唯一通信线程。视觉线程约 30 Hz 更新最新快照，通信线程每 5000 us 读取一次 IMU、复用最新视觉快照并发送完整帧。新增的阶段耗时统计只累加整数，在主线程每秒读取和打印一次，不在 200 Hz 热路径打印。

**Tech Stack:** MaixPy Python、`_thread`、`maix.uart`、Python `unittest`

## Global Constraints

- 只修改 `模型3/maixcam_link.py`、`模型3/main.py` 和 `tests/test_model3_maixcam_link.py`；不改项目目录结构。
- 不修改帧头、协议版本、消息类型、字段顺序、字段宽度、CRC 多项式和 28 字节帧长。
- `MaixCamLink(period_us=5000)` 保持不变；视觉和通信仍是一个主线程加一个现有 `maix.thread.Thread` 工作线程。
- UART 波特率固定为 `460800`；MSPM0 接收端由用户同步配置为 `460800 8N1`。
- 候选矩形可见，但关闭误差文字、误差线和三圆调试绘制，避免显示路径拖慢视觉循环。
- 不触碰用户在任务开始前已有的无关修改；每次只暂存任务明确列出的文件。
- 主机测试只能验证协议、调度和代码契约。最终频率必须用 MaixCAM2 上板日志确认，不能把主机测试等同于硬件实测。

---

### Task 1: 用 CRC 查表替换逐位计算

**Files:**
- Modify: `tests/test_model3_maixcam_link.py`
- Modify: `模型3/maixcam_link.py`
- Test: `tests/test_model3_maixcam_link.py`

**Interfaces:**
- Preserves: `crc16_ccitt_false(data) -> int`
- Adds internal constant: `_CRC16_TABLE: tuple[int, ...]`，长度固定为 256

- [ ] **Step 1: 写入会因查找表缺失而失败的测试**

在 `tests/test_model3_maixcam_link.py` 中增加独立慢速参考实现和测试：

```python
def reference_crc16_ccitt_false(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


class Model3MaixCamLinkTests(unittest.TestCase):
    # 保留现有测试。

    def test_crc_uses_256_entry_table_and_matches_reference(self):
        link = load_link_module()

        self.assertEqual(len(link._CRC16_TABLE), 256)
        for data in (b"", b"123456789", bytes(range(256)), b"\xA5\x5A\x01\x03"):
            self.assertEqual(
                link.crc16_ccitt_false(data),
                reference_crc16_ccitt_false(data),
            )
```

- [ ] **Step 2: 运行定向测试并确认预期失败**

Run: `python -m unittest tests.test_model3_maixcam_link.Model3MaixCamLinkTests.test_crc_uses_256_entry_table_and_matches_reference -v`

Expected: ERROR，指出 `maixcam_link` 没有 `_CRC16_TABLE`；这证明测试确实约束了新的查表实现。

- [ ] **Step 3: 在模块加载时生成 256 项 CRC 表并替换热路径算法**

在 `模型3/maixcam_link.py` 中用以下代码替换现有逐位 CRC 函数：

```python
def _build_crc16_table():
    table = []
    for byte in range(256):
        crc = byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
        table.append(crc)
    return tuple(table)


_CRC16_TABLE = _build_crc16_table()


def crc16_ccitt_false(data):
    crc = 0xFFFF
    table = _CRC16_TABLE
    for byte in data:
        crc = ((crc << 8) ^ table[((crc >> 8) ^ byte) & 0xFF]) & 0xFFFF
    return crc
```

- [ ] **Step 4: 运行 CRC 测试和现有协议测试**

Run: `python -m unittest tests.test_model3_maixcam_link.Model3MaixCamLinkTests.test_crc_uses_256_entry_table_and_matches_reference tests.test_model3_maixcam_link.Model3MaixCamLinkTests.test_crc16_ccitt_false_standard_vector tests.test_model3_maixcam_link.Model3MaixCamLinkTests.test_frame_matches_mspm0_layout_and_crc -v`

Expected: 3 tests PASS，标准向量仍为 `0x29B1`，帧布局和 CRC 字段不变。

- [ ] **Step 5: 精确暂存并中文提交**

```bash
git add "模型3/maixcam_link.py" "tests/test_model3_maixcam_link.py"
git commit -m "性能：加速模型3通信帧校验"
```

---

### Task 2: 增加 200 Hz 热路径的低开销阶段统计

**Files:**
- Modify: `tests/test_model3_maixcam_link.py`
- Modify: `模型3/maixcam_link.py`
- Test: `tests/test_model3_maixcam_link.py`

**Interfaces:**
- Adds: `MaixCamLink.get_timing_stats() -> tuple[int, ...]`
- Tuple fields: `(cycle_count, late_total_us, late_max_us, imu_total_us, imu_max_us, build_total_us, build_max_us, write_total_us, write_max_us, loop_total_us, loop_max_us)`
- Preserves: `get_stats() -> (sent_count, write_error_count, skipped_slot_count)` 和 `get_imu_stats() -> (sample_count, error_count, calibrated)`

- [ ] **Step 1: 写入初始状态和单周期计时测试**

在现有测试类中增加：

```python
    def test_timing_stats_start_at_zero(self):
        link = load_link_module()
        instance = link.MaixCamLink(attitude_source=None)

        self.assertEqual(instance.get_timing_stats(), (0,) * 11)

    def test_worker_records_stage_and_loop_timing(self):
        link = load_link_module()

        class SequencedTime:
            def __init__(self):
                self.values = [1000, 1000, 1005, 1010, 1110, 1210, 1410, 1410]

            def ticks_us(self):
                return self.values.pop(0)

            @staticmethod
            def ticks_ms():
                return 1000

            @staticmethod
            def sleep_us(_duration_us):
                raise AssertionError("worker should not sleep in this fixture")

        class FakeApp:
            def __init__(self):
                self.calls = 0

            def need_exit(self):
                self.calls += 1
                return self.calls > 1

        class FakeSerial:
            def write(self, frame):
                return len(frame)

        source = FakeAttitudeSource((11, -22, 333, -444, 0x06))
        instance = link.MaixCamLink(attitude_source=source)
        instance._app = FakeApp()
        instance._time = SequencedTime()
        instance._serial = FakeSerial()

        instance._tx_worker(None)

        self.assertEqual(
            instance.get_timing_stats(),
            (1, 5, 5, 100, 100, 100, 100, 200, 200, 405, 405),
        )
```

- [ ] **Step 2: 运行两个新测试并确认预期失败**

Run: `python -m unittest tests.test_model3_maixcam_link.Model3MaixCamLinkTests.test_timing_stats_start_at_zero tests.test_model3_maixcam_link.Model3MaixCamLinkTests.test_worker_records_stage_and_loop_timing -v`

Expected: ERROR，指出 `MaixCamLink` 没有 `get_timing_stats()`。

- [ ] **Step 3: 初始化统计并提供只读快照接口**

在 `MaixCamLink.__init__()` 中增加：

```python
        self._timing_stats = (0,) * 11
```

在 `get_stats()` 附近增加：

```python
    def get_timing_stats(self):
        return self._timing_stats
```

- [ ] **Step 4: 在现有 `_tx_worker()` 中累加整数耗时**

保留现有快照解析、IMU 异常处理、整帧构造、UART 异常处理和绝对截止时间推进逻辑，只在相应阶段插入下列计时与汇总结构：

```python
    def _tx_worker(self, _):
        cycle_count = 0
        late_total_us = 0
        late_max_us = 0
        imu_total_us = 0
        imu_max_us = 0
        build_total_us = 0
        build_max_us = 0
        write_total_us = 0
        write_max_us = 0
        loop_total_us = 0
        loop_max_us = 0
        packet_sequence = 0
        sent_count = 0
        write_error_count = 0
        skipped_slot_count = 0
        next_deadline_us = self._time.ticks_us()

        while not self._app.need_exit():
            now_us = self._time.ticks_us()
            if now_us < next_deadline_us:
                self._time.sleep_us(next_deadline_us - now_us)

            cycle_start_us = self._time.ticks_us()
            late_us = max(0, cycle_start_us - next_deadline_us)
            timestamp_ms = self._time.ticks_ms() & 0xFFFFFFFF
            vision_frame, x_error, y_error, flags = resolve_target_snapshot(
                self._target_snapshot,
                timestamp_ms,
                self._target_timeout_ms,
            )

            imu_start_us = self._time.ticks_us()
            (
                yaw_rate_x10,
                pitch_rate_x10,
                yaw_angle_x100,
                pitch_angle_x100,
                imu_flags,
            ) = self._read_attitude_fields()
            imu_end_us = self._time.ticks_us()

            frame = build_frame(
                packet_sequence=packet_sequence,
                vision_frame=vision_frame,
                timestamp_ms=timestamp_ms,
                x_error=x_error,
                y_error=y_error,
                yaw_rate_x10=yaw_rate_x10,
                pitch_rate_x10=pitch_rate_x10,
                yaw_angle_x100=yaw_angle_x100,
                pitch_angle_x100=pitch_angle_x100,
                flags=flags | imu_flags,
            )
            build_end_us = self._time.ticks_us()

            try:
                written = self._serial.write(frame)
            except Exception as exc:
                written = -1
                if write_error_count == 0 or write_error_count % 200 == 0:
                    print("[link] UART4 write exception:", exc)
            write_end_us = self._time.ticks_us()

            sent_count += 1
            packet_sequence = (packet_sequence + 1) & 0xFFFF
            if written != FRAME_SIZE:
                write_error_count += 1
                if write_error_count == 1 or write_error_count % 200 == 0:
                    print("[link] UART4 short write:", written)

            imu_us = imu_end_us - imu_start_us
            build_us = build_end_us - imu_end_us
            write_us = write_end_us - build_end_us
            loop_us = write_end_us - cycle_start_us
            cycle_count += 1
            late_total_us += late_us
            late_max_us = max(late_max_us, late_us)
            imu_total_us += imu_us
            imu_max_us = max(imu_max_us, imu_us)
            build_total_us += build_us
            build_max_us = max(build_max_us, build_us)
            write_total_us += write_us
            write_max_us = max(write_max_us, write_us)
            loop_total_us += loop_us
            loop_max_us = max(loop_max_us, loop_us)
            self._timing_stats = (
                cycle_count,
                late_total_us,
                late_max_us,
                imu_total_us,
                imu_max_us,
                build_total_us,
                build_max_us,
                write_total_us,
                write_max_us,
                loop_total_us,
                loop_max_us,
            )

            next_deadline_us += self._period_us
            now_us = self._time.ticks_us()
            while next_deadline_us <= now_us:
                next_deadline_us += self._period_us
                skipped_slot_count += 1

            self._stats = (sent_count, write_error_count, skipped_slot_count)
```

实施时保持当前 `get_stats()` 三元组和 `_imu_stats` 三元组不变；新增计时使用独立元组，不能把现有调用方改成新的统计格式。

- [ ] **Step 5: 运行通信模块完整测试**

Run: `python -m unittest tests.test_model3_maixcam_link -v`

Expected: 全部 PASS；现有 `get_stats()`、异常回退和绝对 5 ms 截止时间测试没有回归。

- [ ] **Step 6: 精确暂存并中文提交**

```bash
git add "模型3/maixcam_link.py" "tests/test_model3_maixcam_link.py"
git commit -m "性能：统计模型3通信线程耗时"
```

---

### Task 3: 配置 460800、线程切换和低负载可视化

**Files:**
- Modify: `tests/test_model3_maixcam_link.py`
- Modify: `模型3/main.py`
- Test: `tests/test_model3_maixcam_link.py`

**Interfaces:**
- Adds constants: `UART_BAUDRATE = 460800`、`PYTHON_SWITCH_INTERVAL_S = 0.001`、`VISION_YIELD_MS = 1`
- Adds helper: `configure_thread_switching() -> bool`
- Consumes: `MaixCamLink.get_timing_stats()`

- [ ] **Step 1: 写入运行期配置契约测试**

在现有主入口静态测试旁增加：

```python
    def test_main_configures_python_200hz_runtime(self):
        main_path = ROOT / "模型3" / "main.py"
        text = main_path.read_text(encoding="utf-8")
        tree = ast.parse(text)

        required_snippets = (
            "import sys",
            "UART_BAUDRATE = 460800",
            "PYTHON_SWITCH_INTERVAL_S = 0.001",
            "VISION_YIELD_MS = 1",
            "sys.setswitchinterval(PYTHON_SWITCH_INTERVAL_S)",
            "finder.debug_draw_rect = True",
            "finder.debug_draw_err_line = False",
            "finder.debug_draw_err_msg = False",
            "finder.debug_draw_circle = False",
            "baudrate=UART_BAUDRATE",
            "time.sleep_ms(VISION_YIELD_MS)",
            "link.get_timing_stats()",
        )
        for snippet in required_snippets:
            self.assertIn(snippet, text)

        main_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        loops = [node for node in ast.walk(main_function) if isinstance(node, ast.While)]
        self.assertTrue(
            any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "sleep_ms"
                and call.args
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == "VISION_YIELD_MS"
                for loop in loops
                for call in ast.walk(loop)
            )
        )
```

- [ ] **Step 2: 运行新测试并确认预期失败**

Run: `python -m unittest tests.test_model3_maixcam_link.Model3MaixCamLinkTests.test_main_configures_python_200hz_runtime -v`

Expected: FAIL，至少指出当前波特率不是 `460800`，且缺少线程切换、主动让出和耗时统计代码。

- [ ] **Step 3: 增加最小运行期常量和线程切换配置**

在 `模型3/main.py` 顶部增加：

```python
import sys


UART_BAUDRATE = 460800
PYTHON_SWITCH_INTERVAL_S = 0.001
VISION_YIELD_MS = 1


def configure_thread_switching():
    try:
        sys.setswitchinterval(PYTHON_SWITCH_INTERVAL_S)
    except (AttributeError, ValueError) as exc:
        print("[!] Python thread switch interval unchanged:", exc)
        return False
    return True
```

在 `main()` 初始化显示器和模型前调用：

```python
    configure_thread_switching()
```

- [ ] **Step 4: 只保留候选矩形绘制并把 UART4 改为 460800**

在 finder 初始化后明确设置：

```python
    finder.debug_draw_rect = True
    finder.debug_draw_err_line = False
    finder.debug_draw_err_msg = False
    finder.debug_draw_circle = False
```

UART 初始化改为：

```python
    link = MaixCamLink(
        tx_pin="A21",
        device="/dev/ttyS4",
        baudrate=UART_BAUDRATE,
        period_us=5000,
        target_timeout_ms=200,
        attitude_source=attitude,
    )
```

启动日志使用 `UART_BAUDRATE`，避免仍显示旧的 115200。

- [ ] **Step 5: 每帧视觉主动让出 1 ms，并在每秒状态区打印累计阶段统计**

在每次视觉结果发布后、每秒状态判断前增加：

```python
        time.sleep_ms(VISION_YIELD_MS)
```

在现有每秒状态分支内读取：

```python
            (
                timing_count,
                late_total_us,
                late_max_us,
                imu_total_us,
                imu_max_us,
                build_total_us,
                build_max_us,
                write_total_us,
                write_max_us,
                loop_total_us,
                loop_max_us,
            ) = link.get_timing_stats()
            timing_denominator = max(1, timing_count)
            print(
                "[timing] late={:.0f}/{}us imu={:.0f}/{}us "
                "build={:.0f}/{}us write={:.0f}/{}us loop={:.0f}/{}us".format(
                    late_total_us / timing_denominator,
                    late_max_us,
                    imu_total_us / timing_denominator,
                    imu_max_us,
                    build_total_us / timing_denominator,
                    build_max_us,
                    write_total_us / timing_denominator,
                    write_max_us,
                    loop_total_us / timing_denominator,
                    loop_max_us,
                )
            )
```

第一列为累计平均值，第二列为启动以来最大值；不在通信线程内格式化或打印字符串。

- [ ] **Step 6: 运行主入口和通信模块测试**

Run: `python -m unittest tests.test_model3_maixcam_link -v`

Expected: 全部 PASS；主入口仍保持单个通信工作线程，且视觉快照继续由主线程发布。

- [ ] **Step 7: 精确暂存并中文提交**

```bash
git add "模型3/main.py" "tests/test_model3_maixcam_link.py"
git commit -m "性能：配置模型3纯Python 200Hz运行"
```

---

### Task 4: 完成主机回归和 MaixCAM2 上板验收

**Files:**
- Verify: `模型3/main.py`
- Verify: `模型3/maixcam_link.py`
- Verify: `模型3/find_circle.py`
- Test: `tests/test_model3_maixcam_link.py`

**Interfaces:**
- Preserves: 28 字节 UART 帧协议
- Produces: 每秒 `[status]` 与 `[timing]` 两类诊断日志

- [ ] **Step 1: 运行仓库完整单元测试**

Run: `python -m unittest discover -s tests -v`

Expected: 全部测试 PASS。

- [ ] **Step 2: 检查 Python 语法和补丁格式**

Run: `python -m py_compile "模型3/main.py" "模型3/maixcam_link.py" "tests/test_model3_maixcam_link.py"`

Expected: exit code 0。

Run: `git diff --check`

Expected: exit code 0，无尾随空格或冲突标记。

- [ ] **Step 3: 核对协议和修改范围**

Run: `rg -n "FRAME_SIZE|FRAME_HEADER|FRAME_VERSION|_FRAME_FORMAT|period_us=5000|struct\.pack" "模型3/maixcam_link.py" "模型3/main.py"`

Expected: `FRAME_SIZE` 仍为 28，构造参数 `period_us` 仍为 5000，帧头、版本和结构格式与任务开始前一致。

Run: `git status --short --branch`

Expected: 本计划产生的实现和测试均已提交；若仍有修改，只能是任务开始前已有且未纳入本计划的用户文件。

- [ ] **Step 4: 同步 MSPM0 并进行 60 秒上板测试**

1. 将 MSPM0 UART 接收端设置为 `460800 8N1`，协议解析保持不变。
2. 在 MaixCAM2 运行 `模型3/main.py`，保持真实模型推理、显示和串口连接持续 60 秒。
3. 观察相机显示中是否能看到候选矩形；识别有效时确认目标快照持续更新。
4. 保存 60 秒内的 `[status]` 和 `[timing]` 日志。

验收标准：

- `vision` 不低于 30 FPS。
- `tx` 和 `imu` 均稳定在 195–205 Hz。
- `imu_err` 与 `write_err` 不增长。
- `skipped` 的增量不超过 5 次/秒；理想值为 0。
- 候选矩形可见，且没有误差文字、误差线和三圆调试绘制造成的额外负载。

如果仍低于 195 Hz，先按 `[timing]` 的最大耗时定位：`write` 高说明串口驱动/接收端背压，`imu` 高说明传感器读取，`build` 高说明 Python 构帧/CRC，`late` 高但各阶段不高说明主线程仍长时间持有 GIL。不要在没有这组证据前继续盲目调参数。
