# Model 3 IMU Attitude and Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a guarded standalone MaixCAM2 gyro calibration tool and feed calibrated 200 Hz Yaw/Pitch IMU data into Model 3's existing 28-byte UART frame without changing the visual algorithm or frame layout.

**Architecture:** `imu_calibration.py` is the only code allowed to create or overwrite the saved gyro bias and is disabled by default. `imu_attitude.py` loads that bias and exposes one synchronous sample interface; the existing UART worker owns the 5 ms cadence and merges each IMU sample with the latest asynchronous visual snapshot.

**Tech Stack:** MaixPy, `maix.ext_dev.imu`, `maix.ahrs.MahonyAHRS`, Python `unittest`, `ast`, `struct`.

## Global Constraints

- Do not change the Model 3 YOLO model, `find_circle.py` algorithm, camera resolution, or display path.
- Keep `FRAME_VERSION = 0x01`, `FRAME_SIZE = 28`, `_FRAME_FORMAT = "<BBBBHHIhhhhhhH"`, and CRC16 behavior unchanged.
- `CALIBRATION_ENABLE` defaults to `False`; the disabled calibration path must not import `maix`, open the IMU, sample data, or write calibration state.
- Normal runtime may load calibration but must never call `calib_gyro()` or `save_calib_gyro()`.
- Keep the existing `MaixCamLink` arguments backward compatible when `attitude_source=None`, and keep `get_stats()` returning exactly three values.
- Use `save_id="model3_gimbal"`, 10,000 ms gyro calibration, ±4 g acceleration, ±1000 °/s gyro, and 416 Hz ODR for both sensors.
- Treat Yaw angle as relative/debug information; the control-facing stable quantity is Yaw rate.
- Preserve the user's existing uncommitted Chinese comment changes in `模型3/main.py` and `模型3/maixcam_link.py`.

---

### Task 1: Standalone Guarded Calibration Tool

**Files:**
- Create: `模型3/imu_calibration.py`
- Create: `tests/test_model3_imu_calibration.py`

**Interfaces:**
- Consumes: MaixPy `imu.IMU`, `calib_gyro(10000, save_id="model3_gimbal")`.
- Produces: `main() -> int`, `CALIBRATION_ENABLE`, `CALIBRATION_TIME_MS`, and `CALIBRATION_SAVE_ID`.

- [ ] **Step 1: Write the failing calibration guard tests**

```python
def test_calibration_is_disabled_by_default_and_disabled_path_is_safe():
    module = load_module()
    self.assertFalse(module.CALIBRATION_ENABLE)
    self.assertNotIn("maix", sys.modules)
    with mock.patch.object(module, "CALIBRATION_ENABLE", False):
        self.assertEqual(module.main(), 0)
    self.assertNotIn("maix", sys.modules)

def test_enabled_path_uses_fixed_profile_and_saves_bias():
    fake_imu_module = make_fake_imu_module()
    fake_time = types.SimpleNamespace(sleep_ms=mock.Mock())
    with install_fake_maix(fake_imu_module, fake_time):
        module = load_module()
        module.CALIBRATION_ENABLE = True
        self.assertEqual(module.main(), 0)
    fake_imu_module.sensor.calib_gyro.assert_called_once_with(
        10000, save_id="model3_gimbal"
    )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m unittest tests.test_model3_imu_calibration -v`

Expected: FAIL because `模型3/imu_calibration.py` does not exist.

- [ ] **Step 3: Implement the guarded calibration entrypoint**

```python
CALIBRATION_ENABLE = False
CALIBRATION_TIME_MS = 10000
CALIBRATION_SAVE_ID = "model3_gimbal"

def _create_sensor(imu):
    return imu.IMU(
        "default",
        mode=imu.Mode.DUAL,
        acc_scale=imu.AccScale.ACC_SCALE_4G,
        acc_odr=imu.AccOdr.ACC_ODR_416,
        gyro_scale=imu.GyroScale.GYRO_SCALE_1000DPS,
        gyro_odr=imu.GyroOdr.GYRO_ODR_416,
    )

def main():
    if not CALIBRATION_ENABLE:
        print("[imu-cal] disabled; no calibration data was changed")
        return 0
    from maix import time
    from maix.ext_dev import imu
    sensor = _create_sensor(imu)
    print("[imu-cal] keep MaixCAM2, chassis and gimbal still")
    time.sleep_ms(3000)
    bias = sensor.calib_gyro(
        CALIBRATION_TIME_MS, save_id=CALIBRATION_SAVE_ID
    )
    print("[imu-cal] saved bias: {:.6f}, {:.6f}, {:.6f}".format(
        bias.x, bias.y, bias.z
    ))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the focused tests and syntax check**

Run:

```powershell
python -m unittest tests.test_model3_imu_calibration -v
python -m py_compile '模型3/imu_calibration.py' 'tests/test_model3_imu_calibration.py'
```

Expected: all tests pass and both files compile.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- '模型3/imu_calibration.py' 'tests/test_model3_imu_calibration.py'
git commit -m "feat: add guarded model3 imu calibration"
```

---

### Task 2: Calibrated Runtime Attitude Source

**Files:**
- Create: `模型3/imu_attitude.py`
- Create: `tests/test_model3_imu_attitude.py`

**Interfaces:**
- Consumes: `imu.IMU`, `ahrs.MahonyAHRS`, `time.ticks_s()`.
- Produces: `ImuAttitude.start() -> bool`, `ImuAttitude.sample() -> tuple[int, int, int, int, int]`, `ImuAttitude.is_calibrated() -> bool`.
- Tuple order: `(yaw_rate_x10, pitch_rate_x10, yaw_angle_x100, pitch_angle_x100, flags)`.

- [ ] **Step 1: Write failing runtime behavior tests**

```python
def test_missing_calibration_returns_zero_invalid_snapshot():
    source = make_source(calibration_exists=False)
    self.assertFalse(source.start())
    self.assertEqual(source.sample(), (0, 0, 0, 0, 0))

def test_valid_sample_maps_axes_and_scales_fields():
    source = make_source(calibration_exists=True, settle_samples=1)
    self.assertTrue(source.start())
    source.fake_time.values = [1.000, 1.005]
    sample = source.sample()
    self.assertEqual(sample[0], 300)       # gyro.z = 30 deg/s
    self.assertEqual(sample[1], -125)      # gyro.x = -12.5 deg/s
    self.assertEqual(sample[2], 4500)      # angle.z = 45 deg
    self.assertEqual(sample[3], -1000)     # angle.x = -10 deg
    self.assertEqual(sample[4], FLAG_IMU_VALID | FLAG_ATTITUDE_VALID)

def test_large_dt_keeps_rate_but_clears_attitude():
    source = make_source(calibration_exists=True, settle_samples=1)
    source.start()
    source.fake_time.values = [1.000, 1.100]
    yaw_rate, pitch_rate, yaw_angle, pitch_angle, flags = source.sample()
    self.assertNotEqual((yaw_rate, pitch_rate), (0, 0))
    self.assertEqual((yaw_angle, pitch_angle), (0, 0))
    self.assertEqual(flags, FLAG_IMU_VALID)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m unittest tests.test_model3_imu_attitude -v`

Expected: FAIL because `模型3/imu_attitude.py` does not exist.

- [ ] **Step 3: Implement the runtime source with injectable dependencies**

```python
FLAG_IMU_VALID = 1 << 1
FLAG_ATTITUDE_VALID = 1 << 2
CALIBRATION_SAVE_ID = "model3_gimbal"
MAX_DT_S = 0.050
ATTITUDE_SETTLE_SAMPLES = 20
YAW_AXIS = "z"
YAW_SIGN = 1.0
PITCH_AXIS = "x"
PITCH_SIGN = 1.0

class ImuAttitude:
    def __init__(self, sensor=None, attitude_filter=None, time_module=None,
                 settle_samples=ATTITUDE_SETTLE_SAMPLES):
        self._sensor = sensor
        self._filter = attitude_filter
        self._time = time_module
        self._settle_samples = int(settle_samples)
        self._sample_count = 0
        self._last_time_s = None
        self._calibrated = False

    def start(self):
        if self._sensor is None:
            from maix import ahrs, time
            from maix.ext_dev import imu
            self._time = time
            self._sensor = imu.IMU(
                "default", mode=imu.Mode.DUAL,
                acc_scale=imu.AccScale.ACC_SCALE_4G,
                acc_odr=imu.AccOdr.ACC_ODR_416,
                gyro_scale=imu.GyroScale.GYRO_SCALE_1000DPS,
                gyro_odr=imu.GyroOdr.GYRO_ODR_416,
            )
            self._filter = ahrs.MahonyAHRS(2.0, 0.01)
        if not self._sensor.calib_gyro_exists(CALIBRATION_SAVE_ID):
            return False
        self._sensor.load_calib_gyro(CALIBRATION_SAVE_ID)
        self._calibrated = True
        self._last_time_s = self._time.ticks_s()
        return True

    def sample(self):
        if not self._calibrated:
            return (0, 0, 0, 0, 0)
        data = self._sensor.read_all(calib_gryo=True, radian=True)
        now_s = self._time.ticks_s()
        dt = now_s - self._last_time_s
        self._last_time_s = now_s
        rad2deg = 57.29577951308232
        yaw_rate = getattr(data.gyro, YAW_AXIS) * rad2deg * YAW_SIGN
        pitch_rate = getattr(data.gyro, PITCH_AXIS) * rad2deg * PITCH_SIGN
        if dt <= 0 or dt > MAX_DT_S:
            return (round(yaw_rate * 10), round(pitch_rate * 10), 0, 0,
                    FLAG_IMU_VALID)
        angle = self._filter.get_angle(
            data.acc, data.gyro, data.mag, dt, radian=False
        )
        self._sample_count += 1
        flags = FLAG_IMU_VALID
        if self._sample_count >= self._settle_samples:
            flags |= FLAG_ATTITUDE_VALID
        return (round(yaw_rate * 10), round(pitch_rate * 10),
                round(angle.z * YAW_SIGN * 100),
                round(angle.x * PITCH_SIGN * 100), flags)

    def is_calibrated(self):
        return self._calibrated
```

- [ ] **Step 4: Run runtime tests and syntax checks**

Run:

```powershell
python -m unittest tests.test_model3_imu_attitude -v
python -m py_compile '模型3/imu_attitude.py' 'tests/test_model3_imu_attitude.py'
```

Expected: all tests pass and both files compile.

- [ ] **Step 5: Commit Task 2**

```powershell
git add -- '模型3/imu_attitude.py' 'tests/test_model3_imu_attitude.py'
git commit -m "feat: add model3 imu attitude source"
```

---

### Task 3: Merge IMU Samples into the Existing UART Worker

**Files:**
- Modify: `模型3/maixcam_link.py`
- Modify: `tests/test_model3_maixcam_link.py`

**Interfaces:**
- Consumes: optional source with `start() -> bool` and `sample() -> five-int tuple`.
- Produces: unchanged 28-byte frames, `get_imu_stats() -> tuple[int, int, bool]`.

- [ ] **Step 1: Add failing link integration tests**

```python
def test_optional_attitude_source_is_backward_compatible(self):
    link = load_link_module()
    instance = link.MaixCamLink(attitude_source=None)
    self.assertEqual(instance.get_stats(), (0, 0, 0))
    self.assertEqual(instance.get_imu_stats(), (0, 0, False))

def test_attitude_snapshot_is_merged_with_target_flags(self):
    link = load_link_module()
    source = FakeAttitudeSource((111, -222, 3333, -4444,
                                 link.FLAG_IMU_VALID |
                                 link.FLAG_ATTITUDE_VALID))
    instance = link.MaixCamLink(attitude_source=source)
    values = instance._read_attitude_fields()
    self.assertEqual(values, (111, -222, 3333, -4444,
                              link.FLAG_IMU_VALID |
                              link.FLAG_ATTITUDE_VALID))

def test_attitude_exception_degrades_only_imu(self):
    link = load_link_module()
    instance = link.MaixCamLink(attitude_source=FailingAttitudeSource())
    self.assertEqual(instance._read_attitude_fields(), (0, 0, 0, 0, 0))
    self.assertEqual(instance.get_imu_stats(), (0, 1, False))
```

- [ ] **Step 2: Run the existing link test and verify RED**

Run: `python -m unittest tests.test_model3_maixcam_link -v`

Expected: FAIL because `attitude_source` and `get_imu_stats()` do not exist.

- [ ] **Step 3: Add the optional source without changing frame layout**

```python
def __init__(
    self,
    tx_pin="A21",
    device="/dev/ttyS4",
    baudrate=115200,
    period_us=5000,
    target_timeout_ms=200,
    attitude_source=None,
):
    self._tx_pin = tx_pin
    self._device = device
    self._baudrate = baudrate
    self._period_us = period_us
    self._target_timeout_ms = target_timeout_ms
    self._attitude_source = attitude_source
    self._imu_stats = (0, 0, False)

def get_imu_stats(self):
    return self._imu_stats

def _read_attitude_fields(self):
    if self._attitude_source is None:
        return (0, 0, 0, 0, 0)
    samples, errors, calibrated = self._imu_stats
    try:
        fields = self._attitude_source.sample()
        samples += 1
        calibrated = self._attitude_source.is_calibrated()
        self._imu_stats = (samples, errors, calibrated)
        return fields
    except Exception as exc:
        errors += 1
        self._imu_stats = (samples, errors, calibrated)
        if errors == 1 or errors % 200 == 0:
            print("[imu] sample exception:", exc)
        return (0, 0, 0, 0, 0)
```

Add this guarded initialization in `start()` before detaching the worker:

```python
if self._attitude_source is not None:
    try:
        calibrated = bool(self._attitude_source.start())
        self._imu_stats = (0, 0, calibrated)
    except Exception as exc:
        self._imu_stats = (0, 1, False)
        print("[imu] initialization failed:", exc)
```

In `_tx_worker()`, merge the sample without changing the frame layout:

```python
yaw_rate_x10, pitch_rate_x10, yaw_angle_x100, pitch_angle_x100, imu_flags = (
    self._read_attitude_fields()
)
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
```

- [ ] **Step 4: Run link regression tests**

Run:

```powershell
python -m unittest tests.test_model3_maixcam_link -v
python -m py_compile '模型3/maixcam_link.py' 'tests/test_model3_maixcam_link.py'
```

Expected: CRC, frame size, timeout, saturation, main structure, and new IMU tests all pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add -- '模型3/maixcam_link.py' 'tests/test_model3_maixcam_link.py'
git commit -m "feat: send model3 imu data at 200 hz"
```

---

### Task 4: Wire Runtime Attitude into Model 3 Main

**Files:**
- Modify: `模型3/main.py`
- Modify: `tests/test_model3_maixcam_link.py`

**Interfaces:**
- Consumes: `ImuAttitude()` and `MaixCamLink(attitude_source=attitude)`.
- Produces: normal Model 3 application with visual FPS, UART Hz, IMU samples/errors, and calibration status reporting.

- [ ] **Step 1: Extend the main static contract test**

```python
self.assertIn("from imu_attitude import ImuAttitude", text)
self.assertIn("attitude = ImuAttitude()", text)
self.assertIn("attitude_source=attitude", text)
self.assertIn("link.get_imu_stats()", text)
self.assertNotIn("calib_gyro(", text)
self.assertNotIn("save_calib_gyro(", text)
```

- [ ] **Step 2: Run the main contract test and verify RED**

Run: `python -m unittest tests.test_model3_maixcam_link.Model3MaixCamLinkTests.test_main_uses_find_circle_and_publishes_only_updated_results -v`

Expected: FAIL because `main.py` does not import or create `ImuAttitude`.

- [ ] **Step 3: Wire the source and status counters into main**

```python
from imu_attitude import ImuAttitude

attitude = ImuAttitude()
link = MaixCamLink(
    tx_pin="A21",
    device="/dev/ttyS4",
    baudrate=115200,
    period_us=5000,
    target_timeout_ms=200,
    attitude_source=attitude,
)
```

During the existing one-second report, read `imu_samples, imu_errors, imu_calibrated = link.get_imu_stats()` and append those values to the existing status line. Do not move `finder.run()` or change the `if updated:` guard.

- [ ] **Step 4: Run the complete host verification suite**

Run:

```powershell
python -m unittest discover -s tests -v
python -m py_compile '模型3/main.py' '模型3/find_circle.py' '模型3/maixcam_link.py' '模型3/imu_attitude.py' '模型3/imu_calibration.py'
git diff --check
```

Expected: all tests pass, all Model 3 Python files compile, and `git diff --check` exits 0.

- [ ] **Step 5: Audit scope and commit Task 4**

Confirm with `git diff --stat` and `git status --short` that only the planned Model 3 source/tests/plan changed and the pre-existing Chinese comment edits remain preserved. Then run:

```powershell
git add -- '模型3/main.py' 'tests/test_model3_maixcam_link.py'
git commit -m "feat: enable model3 imu attitude runtime"
```

---

### Task 5: Completion Audit and Board Handoff

**Files:**
- Verify only; no source change required.

**Interfaces:**
- Consumes: the design requirements, commits from Tasks 1-4, and host test output.
- Produces: a requirement-by-requirement completion report and explicit MaixCAM2-only verification boundary.

- [ ] **Step 1: Re-run fresh verification**

Run:

```powershell
python -m unittest discover -s tests -v
python -m py_compile '模型3/main.py' '模型3/find_circle.py' '模型3/maixcam_link.py' '模型3/imu_attitude.py' '模型3/imu_calibration.py'
git diff --check
git status --short
```

Expected: tests and compilation pass; only unrelated pre-existing worktree changes, if any, remain.

- [ ] **Step 2: Check every fixed contract**

Run:

```powershell
rg -n "CALIBRATION_ENABLE = False|CALIBRATION_TIME_MS = 10000|model3_gimbal" '模型3/imu_calibration.py' '模型3/imu_attitude.py'
rg -n "FRAME_VERSION = 0x01|FRAME_SIZE = 28|_FRAME_FORMAT" '模型3/maixcam_link.py'
rg -n "ImuAttitude|attitude_source|get_imu_stats" '模型3/main.py' '模型3/maixcam_link.py'
```

Expected: all fixed values and integration points are present in their planned files.

- [ ] **Step 3: Report the hardware verification boundary**

State explicitly that host evidence proves syntax, calibration guard behavior, runtime degradation, scaling, flags, 28-byte layout, CRC, and main wiring. Do not claim that LSM6DSOWTR axis direction, sensor availability, 200 Hz hardware cadence, five-minute drift, or thermal/visual performance has passed until the MaixCAM2 board tests in the design document are run.
