# MaixCAM2 Find Circle Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone `Maixcam2/find_circle.py` that preserves the 2025 vision algorithm while loading the already working MaixCAM2 model.

**Architecture:** Copy the proven `FindRectCircle` implementation without restructuring its OpenCV pipeline. Adapt only model discovery/loading for `model_246619.mud`; keep `Maixcam2/main.py` untouched and retain the module's standalone loop.

**Tech Stack:** MaixPy (`maix.camera`, `display`, `image`, `nn`, `app`, `time`), OpenCV, NumPy, Python standard-library `unittest` and `ast`.

## Global Constraints

- Create `Maixcam2/find_circle.py`; do not modify `Maixcam2/main.py`.
- Preserve `FindRectCircle(disp)`, `get_res()`, `run()`, and the five-item return contract.
- Preserve the existing 448 x 448 and OpenCV processing pipeline.
- Use `/root/models/_model25e_maixcam2/model_246619.mud` first and the script-local `model_246619.mud` as fallback.
- Do not add communication, motor, laser, or unrelated refactoring.
- Board camera, NPU, display, and frame-rate behavior remain hardware verification items.

---

### Task 1: Port the standalone find-circle module

**Files:**
- Create: `tests/test_maixcam2_find_circle_port.py`
- Create: `Maixcam2/find_circle.py`
- Do not modify: `Maixcam2/main.py`

**Interfaces:**
- Consumes: `display.Display` supplied to `FindRectCircle(disp)` and `model_246619.mud` with its two `.axmodel` files.
- Produces: `FindRectCircle.get_res()` and `FindRectCircle.run() -> [last_center, center_pos, err_center, circle3_points, updated]`.

- [x] **Step 1: Write the failing port-contract test**

```python
import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "Maixcam2" / "find_circle.py"


class FindCirclePortContractTests(unittest.TestCase):
    def test_standalone_maixcam2_port_contract(self):
        self.assertTrue(TARGET.is_file(), "Maixcam2/find_circle.py has not been ported")

        text = TARGET.read_text(encoding="utf-8")
        tree = ast.parse(text)

        self.assertIn("/root/models/_model25e_maixcam2/model_246619.mud", text)
        self.assertIn("model_246619.mud", text)
        self.assertIn("os.path.dirname(os.path.abspath(__file__))", text)
        self.assertIn("nn.YOLOv5(model=model_path", text)
        self.assertNotIn("model_3356.mud", text)

        classes = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
        }
        self.assertIn("FindRectCircle", classes)
        methods = {
            node.name: node
            for node in classes["FindRectCircle"].body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("get_res", methods)
        self.assertIn("run", methods)

        expected_return = [
            "self.last_center",
            "self.center_pos",
            "self.err_center",
            "self.last_circle3_points",
            "self.updated",
        ]
        list_returns = [
            [ast.unparse(item) for item in node.value.elts]
            for node in ast.walk(methods["run"])
            if isinstance(node, ast.Return) and isinstance(node.value, ast.List)
        ]
        self.assertIn(expected_return, list_returns)

        has_main_guard = any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            for node in tree.body
        )
        self.assertTrue(has_main_guard)


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run the test and verify the RED state**

Run:

```powershell
python -m unittest tests.test_maixcam2_find_circle_port -v
```

Expected: `FAIL` with `Maixcam2/find_circle.py has not been ported`.

- [x] **Step 3: Copy the existing algorithm and make the minimal model-loading change**

Create `Maixcam2/find_circle.py` as an exact copy of
`2025校赛视觉/find_circle.py`, then replace only its model path and loading block
with the following code. All other algorithm and standalone-entry code remains
byte-for-byte equivalent apart from line-ending and trailing-whitespace normalization.

```python
    model_path = "/root/models/_model25e_maixcam2/model_246619.mud"

    def __init__(self, disp):
        model_path = self.model_path
        if not os.path.exists(model_path):
            local_model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "model_246619.mud",
            )
            if not os.path.exists(local_model_path):
                print(
                    f"load model failed, please put model in {self.model_path}, "
                    f"or {local_model_path}"
                )
            model_path = local_model_path

        self.disp = disp
        self.detector = nn.YOLOv5(model=model_path, dual_buff=self.model_dual_buff_mode)
```

- [x] **Step 4: Run the contract test and verify the GREEN state**

Run:

```powershell
python -m unittest tests.test_maixcam2_find_circle_port -v
```

Expected: one test passes and the command exits with code 0.

- [x] **Step 5: Run syntax and scope verification**

Run:

```powershell
python -m py_compile Maixcam2/find_circle.py tests/test_maixcam2_find_circle_port.py
git diff --check
git diff --exit-code -- Maixcam2/main.py
git diff --no-index -- 2025校赛视觉/find_circle.py Maixcam2/find_circle.py
```

Expected:

- Both Python files compile without output.
- `git diff --check` exits 0.
- `Maixcam2/main.py` has no diff.
- The no-index diff exits 1 only because it shows the intended MaixCAM2 model-loading block; no OpenCV pipeline changes appear.

- [ ] **Step 6: Commit the port**

```powershell
git add -- Maixcam2/find_circle.py tests/test_maixcam2_find_circle_port.py docs/superpowers/plans/2026-07-15-maixcam2-find-circle.md
git commit -m "feat: port find circle to MaixCAM2"
```
