import tempfile
import unittest
from pathlib import Path

import pandas as pd

from examples.view_exp_artifacts import (
    assess_strategy_history,
    artifact_dir,
    artifact_paths,
    build_email_body,
    load_metrics_history,
    load_position_tracking,
    load_sig_analysis,
    latest_run_dir,
    render_metrics_history_chart,
)


class TestViewExpArtifactsMetrics(unittest.TestCase):
    def test_latest_run_uses_mlflow_time_and_skips_failed_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir)
            successful = experiment_dir / "successful"
            failed = experiment_dir / "failed"
            for run_dir, start_time, status in (
                (successful, 100, 3),
                (failed, 200, 4),
            ):
                artifacts = run_dir / "artifacts"
                artifacts.mkdir(parents=True)
                (artifacts / "artifact.txt").write_text("data", encoding="utf-8")
                (run_dir / "meta.yaml").write_text(
                    f"start_time: {start_time}\nstatus: {status}\n",
                    encoding="utf-8",
                )

            self.assertEqual(latest_run_dir(experiment_dir), successful)

    def test_artifact_paths_uses_mlflow_artifact_uri(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "mlruns" / "exp_id" / "run_id"
            recorded_artifacts = root / "external artifacts"
            (run_dir / "artifacts").mkdir(parents=True)
            recorded_artifacts.mkdir()
            (run_dir / "meta.yaml").write_text(
                f"artifact_uri: {recorded_artifacts.as_uri()}\n",
                encoding="utf-8",
            )

            self.assertEqual(artifact_dir(run_dir), recorded_artifacts)
            self.assertEqual(artifact_paths(run_dir)["pred"], recorded_artifacts / "pred.pkl")

    def test_artifact_paths_includes_risk_metrics(self):
        run_dir = Path("mlruns") / "exp_id" / "run_id"

        self.assertEqual(
            artifact_paths(run_dir)["risk_metrics"],
            run_dir / "artifacts" / "risk_metrics.pkl",
        )

    def test_signal_table_and_email_include_risk_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            sig_dir = run_dir / "artifacts" / "sig_analysis"
            sig_dir.mkdir(parents=True)
            pd.Series([0.1, 0.2, 0.3]).to_pickle(sig_dir / "ic.pkl")
            pd.Series([0.2, 0.3, 0.4]).to_pickle(sig_dir / "ric.pkl")
            pd.DataFrame(
                {"value": {"Sharpe": 1.5, "Sortino": 2.0, "Calmar": 1.2}}
            ).to_pickle(run_dir / "artifacts" / "risk_metrics.pkl")

            result = load_sig_analysis(run_dir)

            self.assertIsNotNone(result)
            self.assertEqual(
                result.index.tolist(),
                ["IC", "Rank IC", "ICIR", "Rank ICIR", "Sharpe", "Sortino", "Calmar"],
            )
            frame = pd.DataFrame([{"instrument": "TEST", "action": "HOLD", "score": 1.0}])
            body = build_email_body(run_dir, frame, frame, None, None, sig_analysis=result)
            for metric in result.index:
                self.assertIn(metric, body)

    def test_adaptive_run_loads_five_metrics_without_sig_analysis_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            artifacts = run_dir / "artifacts"
            artifacts.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "trade_date": pd.Timestamp("2026-08-04 16:00:00"),
                        "strategy": "Adaptive_Exit",
                        "IC": 0.03,
                        "ICIR": 0.6,
                        "Sharpe": 1.2,
                        "Sortino": 1.5,
                        "Calmar": 0.9,
                    }
                ]
            ).to_pickle(artifacts / "strategy_metrics.pkl")

            result = load_sig_analysis(run_dir)
            body = build_email_body(run_dir, None, None, None, None, sig_analysis=result)

        self.assertEqual(result.index.tolist(), ["IC", "ICIR", "Sharpe", "Sortino", "Calmar"])
        for metric in ("IC", "ICIR", "Sharpe", "Sortino", "Calmar"):
            self.assertIn(metric, body)

    def test_position_tracking_is_loaded_and_rendered_in_email(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            artifacts = run_dir / "artifacts"
            artifacts.mkdir(parents=True)
            positions = pd.DataFrame(
                [
                    {
                        "strategy": "Adaptive_Exit",
                        "instrument": "SH600000",
                        "entry_date": pd.Timestamp("2026-08-03 09:30:00"),
                        "last_updated": pd.Timestamp("2026-08-04 15:00:00"),
                        "holding_days": 2,
                        "score": 0.91,
                        "rank": 1,
                        "weight": 0.2,
                        "amount": 60000.0,
                    }
                ]
            )
            actions = pd.DataFrame(
                [
                    {
                        "strategy": "Adaptive_Exit",
                        "trade_date": pd.Timestamp("2026-08-04"),
                        "action": "HOLD",
                        "instrument": "SH600000",
                        "holding_days": 2,
                        "reason": "尚未触发动态退出条件",
                    }
                ]
            )
            positions.to_pickle(artifacts / "current_positions.pkl")
            actions.to_pickle(artifacts / "holding_actions.pkl")
            actions.to_pickle(artifacts / "holding_history.pkl")

            tracking = load_position_tracking(run_dir)
            body = build_email_body(
                run_dir,
                None,
                None,
                None,
                None,
                current_positions=tracking["current_positions"],
                holding_actions=tracking["holding_actions"],
            )

            self.assertEqual(tracking["current_positions"].loc[0, "holding_days"], 2)
            self.assertIn("当前策略持仓", body)
            self.assertIn("今日持仓动作", body)
            self.assertIn("SH600000", body)
            self.assertIn("amount", body)
            self.assertIn("60000", body)
            self.assertIn("尚未触发动态退出条件", body)
            self.assertIn("2026-08-03", body)
            self.assertIn("2026-08-04", body)
            self.assertNotIn("09:30:00", body)
            self.assertNotIn("15:00:00", body)
            self.assertNotIn("16:00:00", body)

    def test_old_run_without_position_artifacts_remains_supported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracking = load_position_tracking(Path(temp_dir))

        self.assertEqual(
            tracking,
            {"current_positions": None, "holding_actions": None, "holding_history": None},
        )

    def test_metric_history_chart_and_strategy_advice(self):
        dates = pd.date_range("2026-07-27", periods=6, freq="B")
        rows = []
        for strategy, multiplier in (("1D_Short_Term", 1.0), ("5D_Mid_Term", 0.4)):
            for trade_date in dates:
                rows.append(
                    {
                        "trade_date": trade_date,
                        "strategy": strategy,
                        "IC": 0.04 * multiplier,
                        "ICIR": 0.8 * multiplier,
                        "Sharpe": 1.4 * multiplier,
                        "Sortino": 1.6 * multiplier,
                        "Calmar": 0.9 * multiplier,
                    }
                )
        history = pd.DataFrame(rows)

        advice = assess_strategy_history(history)
        chart = render_metrics_history_chart(history)
        body = build_email_body(
            Path("run"),
            None,
            None,
            None,
            None,
            strategy_advice=advice,
            metrics_chart_cid="strategy-metrics-history",
        )

        statuses = advice.set_index("strategy")["是否有效"].to_dict()
        self.assertEqual(statuses["1D_Short_Term"], "有效")
        self.assertEqual(statuses["5D_Mid_Term"], "无效")
        self.assertEqual(statuses["Adaptive_Exit"], "数据不足")
        priorities = advice.set_index("strategy")["策略优先级"].to_dict()
        self.assertEqual(priorities["1D_Short_Term"], "第 1 优先")
        self.assertEqual(priorities["5D_Mid_Term"], "第 2 优先")
        self.assertEqual(priorities["Adaptive_Exit"], "待定")
        self.assertTrue(chart.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn('cid:strategy-metrics-history', body)
        self.assertIn("三策略有效性与调整建议", body)

    def test_load_metrics_history_merges_and_deduplicates_local_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            artifacts = run_dir / "artifacts"
            artifacts.mkdir(parents=True)
            columns = {
                "trade_date": [pd.Timestamp("2026-08-03")],
                "strategy": ["1D_Short_Term"],
                "IC": [0.03],
                "ICIR": [0.6],
                "Sharpe": [1.1],
                "Sortino": [1.2],
                "Calmar": [0.7],
            }
            pd.DataFrame(columns).to_pickle(artifacts / "metrics_history.pkl")
            shared = pd.DataFrame({**columns, "Sharpe": [1.3]})
            shared_path = root / "shared.pkl"
            shared.to_pickle(shared_path)

            history = load_metrics_history(run_dir, shared_path)

        self.assertEqual(len(history), 1)
        self.assertEqual(history.loc[0, "Sharpe"], 1.3)


if __name__ == "__main__":
    unittest.main()
