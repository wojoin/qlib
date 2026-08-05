import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from examples.view_exp_artifacts import (
    DEFAULT_EXPORT_DIR,
    action_plan,
    artifact_paths,
    assess_strategy_quality,
    build_feature_diagnostics,
    candidate_top10,
    describe_dataframe,
    default_export_dir,
    explain_instrument,
    strategy_kwargs_from_workflow,
)


class TestViewCPOArtifacts(unittest.TestCase):
    def test_artifact_paths_resolves_expected_files_from_run_dir(self):
        run_dir = Path("mlruns") / "exp_id" / "run_id"
        paths = artifact_paths(run_dir)

        self.assertEqual(paths["pred"], run_dir / "artifacts" / "pred.pkl")
        self.assertEqual(
            paths["port_analysis"],
            run_dir / "artifacts" / "portfolio_analysis" / "port_analysis_1day.pkl",
        )
        self.assertEqual(
            paths["indicator_analysis"],
            run_dir / "artifacts" / "portfolio_analysis" / "indicator_analysis_1day.pkl",
        )

    def test_describe_dataframe_includes_shape_columns_head_and_tail(self):
        df = pd.DataFrame({"score": [1.0, 2.0, 3.0]})

        text = describe_dataframe(df, rows=2)

        self.assertIn("shape: (3, 1)", text)
        self.assertIn("columns: score", text)
        self.assertIn("head:", text)
        self.assertIn("tail:", text)

    def test_default_export_dir_root_is_examples_cpo(self):
        self.assertEqual(DEFAULT_EXPORT_DIR, Path("examples") / "cpo")

    def test_default_export_dir_uses_current_date_folder(self):
        self.assertEqual(default_export_dir(date(2026, 5, 6)), Path("examples") / "cpo" / "20260506")

    def test_action_plan_marks_buy_sell_and_hold_for_topk_dropout(self):
        pred = pd.Series(
            {
                "BUY1": 0.50,
                "HOLD1": 0.40,
                "HOLD2": 0.30,
                "HOLD3": 0.20,
                "HOLD4": 0.10,
                "SELL1": -0.10,
            },
            name="score",
        )
        positions = pd.DataFrame(
            {
                "instrument": ["HOLD1", "HOLD2", "HOLD3", "HOLD4", "SELL1"],
                "weight": [0.20, 0.20, 0.20, 0.20, 0.20],
                "amount": [1, 1, 1, 1, 1],
                "price": [10, 10, 10, 10, 10],
                "count_day": [1, 1, 1, 1, 1],
            }
        )

        plan = action_plan(pred, positions, topk=5, n_drop=1)

        actions = dict(zip(plan["instrument"], plan["action"]))
        self.assertEqual(actions["BUY1"], "BUY")
        self.assertEqual(actions["SELL1"], "SELL")
        self.assertEqual(actions["HOLD1"], "HOLD")
        self.assertIn("interpretation", plan.columns)
        self.assertIn("reason", plan.columns)
        self.assertIn("卖出", plan.loc[plan["instrument"] == "SELL1", "interpretation"].iloc[0])
        self.assertIn("新买入", plan.loc[plan["instrument"] == "BUY1", "interpretation"].iloc[0])
        self.assertIn("继续持有", plan.loc[plan["instrument"] == "HOLD1", "interpretation"].iloc[0])
        self.assertIn("评分最低", plan.loc[plan["instrument"] == "SELL1", "reason"].iloc[0])
        self.assertIn("候选标的", plan.loc[plan["instrument"] == "BUY1", "reason"].iloc[0])
        self.assertIn("组合内", plan.loc[plan["instrument"] == "HOLD1", "reason"].iloc[0])

    def test_action_plan_uses_n_drop_two_when_requested(self):
        pred = pd.Series(
            {
                "BUY1": 0.60,
                "BUY2": 0.55,
                "HOLD1": 0.50,
                "HOLD2": 0.40,
                "HOLD3": 0.30,
                "SELL1": -0.10,
                "SELL2": -0.20,
            },
            name="score",
        )
        positions = pd.DataFrame(
            {
                "instrument": ["HOLD1", "HOLD2", "HOLD3", "SELL1", "SELL2"],
                "weight": [0.20, 0.20, 0.20, 0.20, 0.20],
                "amount": [1, 1, 1, 1, 1],
                "price": [10, 10, 10, 10, 10],
                "count_day": [1, 1, 1, 1, 1],
            }
        )

        plan = action_plan(pred, positions, topk=5, n_drop=2)

        self.assertEqual((plan["action"] == "SELL").sum(), 2)
        self.assertEqual((plan["action"] == "BUY").sum(), 2)
        self.assertEqual(set(plan.loc[plan["action"] == "SELL", "instrument"]), {"SELL1", "SELL2"})
        self.assertEqual(set(plan.loc[plan["action"] == "BUY", "instrument"]), {"BUY1", "BUY2"})

    def test_strategy_kwargs_from_workflow_reads_topk_and_n_drop(self):
        kwargs = strategy_kwargs_from_workflow(Path("examples") / "workflow_by_cpo.py")

        self.assertEqual(kwargs["topk"], 5)
        self.assertEqual(kwargs["n_drop"], 2)

    def test_feature_diagnostics_explain_specific_strong_and_weak_features(self):
        pred = pd.Series({"AAA": 0.9, "BBB": -0.2, "CCC": 0.1})
        raw = pd.DataFrame(
            {
                "MA5": [1.10, 0.90, 1.00],
                "MA10": [1.00, 1.00, 1.00],
                "VOL_RATIO20": [2.0, 0.5, 1.0],
            },
            index=["AAA", "BBB", "CCC"],
        )

        diagnostics = build_feature_diagnostics(pred, raw, raw)
        strong_text = explain_instrument("AAA", diagnostics, "support")
        weak_text = explain_instrument("BBB", diagnostics, "drag")

        self.assertIn("MA5", strong_text)
        self.assertIn("放量", strong_text)
        self.assertIn("MA5", weak_text)

    def test_candidate_top10_exports_ranked_candidates_with_reasons(self):
        pred = pd.Series({f"S{i}": 10 - i for i in range(12)})
        raw = pd.DataFrame({"VOL_RATIO20": list(range(12, 0, -1))}, index=[f"S{i}" for i in range(12)])
        diagnostics = build_feature_diagnostics(pred, raw, raw)

        result = candidate_top10(pred, current_holding={"S0"}, diagnostics=diagnostics, limit=10)

        self.assertEqual(len(result), 10)
        self.assertEqual(result.iloc[0]["rank"], 1)
        self.assertIn("feature_reason", result.columns)

    def test_assess_strategy_quality_reports_buy_observe_and_sell_lists(self):
        plan = pd.DataFrame(
            [
                {"action": "BUY", "instrument": "000001", "score": 0.95},
                {"action": "HOLD", "instrument": "000002", "score": 0.70},
                {"action": "SELL", "instrument": "000003", "score": 0.10},
            ]
        )

        summary = assess_strategy_quality(plan)

        self.assertEqual(summary["overall"], "较好")
        self.assertEqual(summary["buy_list"], ["000001"])
        self.assertEqual(summary["observe_list"], ["000002"])
        self.assertEqual(summary["sell_list"], ["000003"])
        self.assertIn("买入", summary["summary_text"])
        self.assertIn("观望", summary["summary_text"])
        self.assertIn("卖出", summary["summary_text"])


if __name__ == "__main__":
    unittest.main()
