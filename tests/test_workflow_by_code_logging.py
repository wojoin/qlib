import ast
from pathlib import Path
import unittest


class TestWorkflowByCodeLogging(unittest.TestCase):
    def test_workflow_configures_file_logging_before_qlib_init(self):
        workflow_path = Path(__file__).parents[1] / "examples" / "workflow_by_code.py"
        source = workflow_path.read_text()
        tree = ast.parse(source)

        self.assertIn("logging_config", source)
        self.assertIn('"file"', source)
        self.assertIn("FileHandler", source)
        self.assertIn("examples/logs", source)

        qlib_init_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "init"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "qlib"
        ]
        self.assertTrue(
            any(any(keyword.arg == "logging_config" for keyword in call.keywords) for call in qlib_init_calls)
        )


if __name__ == "__main__":
    unittest.main()
