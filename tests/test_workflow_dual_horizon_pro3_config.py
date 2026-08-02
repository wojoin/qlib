import ast
from pathlib import Path
import unittest


class TestWorkflowDualHorizonPro3Config(unittest.TestCase):
    def test_robust_zscore_norm_processors_define_fit_window(self):
        workflow_path = Path(__file__).parents[1] / "examples" / "workflow_dual_horizon_pro3.py"
        tree = ast.parse(workflow_path.read_text())

        robust_processors = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            class_value = None
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "class":
                    class_value = value
                    break
            if not (isinstance(class_value, ast.Constant) and class_value.value == "RobustZScoreNorm"):
                continue
            dict_value = ast.literal_eval(node)
            if dict_value.get("class") == "RobustZScoreNorm":
                robust_processors.append(dict_value)

        self.assertGreater(len(robust_processors), 0)
        for processor in robust_processors:
            kwargs = processor["kwargs"]
            self.assertEqual(kwargs["fit_start_time"], "2020-01-01")
            self.assertEqual(kwargs["fit_end_time"], "2024-12-31")


if __name__ == "__main__":
    unittest.main()
