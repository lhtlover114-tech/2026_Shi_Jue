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
