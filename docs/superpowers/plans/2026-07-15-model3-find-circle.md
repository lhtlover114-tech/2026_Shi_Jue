# Model 3 Find Circle Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone `模型3/find_circle.py` that preserves the 2025 OpenCV pipeline while using the proven Model 3 YOLO11 model.

**Architecture:** Copy the existing `FindRectCircle` implementation without restructuring it. Adapt only model discovery/loading and the initial camera-size mode; keep `模型3/main.py` untouched.

**Tech Stack:** MaixPy, YOLO11, OpenCV, NumPy, Python `unittest` and `ast`.

## Global Constraints

- Do not modify `模型3/main.py`.
- Preserve `FindRectCircle(disp)`, `get_res()`, and the five-item `run()` return contract.
- Use `/root/models/_model25e_maixcam2/best.mud` with a script-local `best.mud` fallback.
- Use `nn.YOLO11` and remove all active references to YOLOv5 and old model files.
- Set `hires_mode = False` for the first board test.
- Do not refactor the OpenCV processing pipeline or add control/communication logic.

---

### Task 1: Port the standalone module

**Files:**
- Create: `tests/test_model3_find_circle_port.py`
- Create: `模型3/find_circle.py`
- Do not modify: `模型3/main.py`

**Interfaces:**
- Consumes: `display.Display`, `best.mud`, `best_npu.axmodel`, and `best_vnpu.axmodel`.
- Produces: `FindRectCircle.get_res()` and `FindRectCircle.run() -> [last_center, center_pos, err_center, circle3_points, updated]`.

- [x] **Step 1: Write the failing static contract test**

  The test must require the target file, YOLO11 loader, absolute and local model
  paths, `hires_mode = False`, the class methods, and the five-item return value.

- [x] **Step 2: Run the test and verify RED**

  Run: `python -m unittest tests.test_model3_find_circle_port -v`

  Expected: failure because `模型3/find_circle.py` does not exist.

- [x] **Step 3: Copy the 2025 implementation and apply the minimal adaptation**

  Use the original OpenCV pipeline unchanged. Replace the old model-loading block
  with `best.mud` path resolution and `nn.YOLO11(model=model_path, ...)`, and set
  `hires_mode = False`.

- [x] **Step 4: Run the test and syntax checks**

  Run:

  ```powershell
  python -m unittest tests.test_model3_find_circle_port -v
  python -m py_compile '模型3/find_circle.py' 'tests/test_model3_find_circle_port.py'
  ```

  Expected: both commands exit 0.

- [x] **Step 5: Verify scope and source preservation**

  Compare the pre/post SHA-256 of `模型3/main.py`; compare the source and target
  files to confirm only the planned model-loading and `hires_mode` changes exist.
