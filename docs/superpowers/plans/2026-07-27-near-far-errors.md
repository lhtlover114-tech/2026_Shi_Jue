# Near/Far Errors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independently filtered near-field and far-field line errors and transmit them in the existing UART `x_error` and `y_error` fields.

**Architecture:** Put hardware-independent region math in `巡线/line_metrics.py`, call it from `LineFollower.process()`, and preserve the current overall error/confidence path. Update `main.py` and `uart_link.py` so the two values are published without changing the 32-byte protocol.

**Tech Stack:** Python 3 / MaixPy v4, `unittest`, existing `struct`-based UART protocol.

## Global Constraints

- Preserve `FRAME_VERSION = 0x02` and `FRAME_SIZE = 32`.
- The top five configured strips are the far region.
- The bottom five configured strips are the near region.
- Positive error remains right of image center; negative remains left.
- Keep existing `error` and `confidence` output keys.

---

### Task 1: Region error math

**Files:**
- Create: `巡线/line_metrics.py`
- Create: `巡线/tests/test_line_metrics.py`

**Interfaces:**
- Produces: `compute_region_raw_errors(centers, center_x, strip_y, region_strip_count) -> (near_error_or_none, far_error_or_none)`.

- [ ] Write tests for fixed top/bottom region selection, weighted means, sign, and absent regions.
- [ ] Run `python -m unittest discover -s 巡线/tests -v` and verify the new tests fail because the helper is missing.
- [ ] Implement the minimal helper.
- [ ] Run the same test command and verify all tests pass.

### Task 2: Vision result integration

**Files:**
- Modify: `巡线/find_line.py`

**Interfaces:**
- Consumes: `compute_region_raw_errors(...)`.
- Produces: result keys `near_error` and `far_error`, both floats in image pixels.

- [ ] Add `REGION_STRIPS = 5` and independent last-error state.
- [ ] Compute and independently low-pass-filter near/far errors.
- [ ] Preserve the existing overall `error`, `center_x`, and `confidence` behavior.
- [ ] Add near/far values to debug text and returned result.
- [ ] Run unit tests and `python -m py_compile` with a temporary stub `maix` module.

### Task 3: UART field mapping

**Files:**
- Modify: `巡线/main.py`
- Modify: `巡线/uart_link.py`

**Interfaces:**
- `publish_line_data(near_error, confidence=1.0, fps=0.0, far_error=0.0)`.
- UART `x_error = near_error`, `y_error = far_error`.

- [ ] Update the publisher call using explicit keyword arguments.
- [ ] Store both errors in the transmission snapshot.
- [ ] Rename debug labels to `near` and `far`.
- [ ] Verify frame size remains 32 bytes and both signed int16 fields unpack correctly.
- [ ] Run the complete test and syntax verification commands.

### Task 4: Review

**Files:**
- Review all changed files.

- [ ] Compare branch against `9dd25ef4ff71e49b6313df06f1fc4df52bc197a3`.
- [ ] Confirm no unrelated behavior or protocol layout changed.
- [ ] Open a draft pull request with testing evidence and MCU field mapping notes.
