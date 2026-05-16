import ast
from pathlib import Path
import unittest


class TestWorkflowByCPOConfig(unittest.TestCase):
    def test_cpo_task_uses_data_handler_with_qlib_data_loader_config(self):
        workflow_path = Path(__file__).parents[1] / "examples" / "workflow_by_cpo.py"
        tree = ast.parse(workflow_path.read_text())

        cpo_task = next(
            node.value for node in tree.body if isinstance(node, ast.Assign) and node.targets[0].id == "CPO_TASK"
        )
        config = ast.literal_eval(cpo_task)

        handler_config = config["dataset"]["kwargs"]["handler"]
        self.assertEqual(handler_config["class"], "DataHandlerLP")
        self.assertEqual(handler_config["module_path"], "qlib.data.dataset.handler")
        self.assertEqual(handler_config["kwargs"]["instruments"], "cpo_sector")
        self.assertEqual(handler_config["kwargs"]["data_loader"]["class"], "QlibDataLoader")
        self.assertNotIn("feature", handler_config["kwargs"])


if __name__ == "__main__":
    unittest.main()
