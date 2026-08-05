import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from examples.workflow_dual_horizon_pro2 import (
    calculate_risk_metrics,
    calculate_strategy_metrics,
    evaluate_adaptive_history,
    persist_metrics_history,
)


class TestWorkflowDualHorizonPro2Metrics(unittest.TestCase):
    def test_calculate_risk_metrics_uses_returns_net_of_cost(self):
        report = pd.DataFrame(
            {
                "return": [0.02, -0.01, 0.03, -0.02],
                "cost": [0.001, 0.001, 0.001, 0.001],
            }
        )
        net_returns = report["return"] - report["cost"]

        metrics = calculate_risk_metrics(report, annualization_factor=4)

        expected_sharpe = net_returns.mean() / net_returns.std(ddof=1) * np.sqrt(4)
        downside = np.sqrt(np.square(net_returns.clip(upper=0.0)).mean())
        expected_sortino = net_returns.mean() / downside * np.sqrt(4)
        equity = (1.0 + net_returns).cumprod()
        max_drawdown = (equity / equity.cummax() - 1.0).min()
        expected_calmar = (equity.iloc[-1] - 1.0) / abs(max_drawdown)

        self.assertAlmostEqual(metrics.loc["Sharpe", "value"], expected_sharpe)
        self.assertAlmostEqual(metrics.loc["Sortino", "value"], expected_sortino)
        self.assertAlmostEqual(metrics.loc["Calmar", "value"], expected_calmar)

    def test_calculate_risk_metrics_requires_return_and_cost(self):
        with self.assertRaisesRegex(ValueError, "cost"):
            calculate_risk_metrics(pd.DataFrame({"return": [0.01]}))

    def test_metric_history_replaces_same_strategy_and_trade_date(self):
        risk = pd.DataFrame({"value": {"Sharpe": 1.2, "Sortino": 1.4, "Calmar": 0.8}})
        first = calculate_strategy_metrics(
            pd.Series([0.01, 0.03, 0.05]), risk, "1D_Short_Term", "2026-08-03"
        )
        updated_risk = pd.DataFrame({"value": {"Sharpe": 1.5, "Sortino": 1.7, "Calmar": 1.1}})
        updated = calculate_strategy_metrics(
            pd.Series([0.02, 0.04, 0.06]), updated_risk, "1D_Short_Term", "2026-08-03"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics_history.pkl"
            persist_metrics_history(first, path)
            history = persist_metrics_history(updated, path)

        self.assertEqual(len(history), 1)
        self.assertEqual(history.loc[0, "Sharpe"], 1.5)
        self.assertEqual(
            history.columns.tolist(),
            ["trade_date", "strategy", "IC", "ICIR", "Sharpe", "Sortino", "Calmar"],
        )

    def test_adaptive_history_produces_ic_report_and_risk_metrics(self):
        dates = pd.date_range("2026-07-27", periods=8, freq="B")
        instruments = ["AAA", "BBB", "CCC", "DDD"]
        index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        scores = pd.Series(
            [0.9, 0.7, 0.3, 0.1] * len(dates),
            index=index,
            name="score",
        )
        returns = pd.Series(
            [0.02, 0.01, -0.01, -0.02] * len(dates),
            index=index,
            name="forward_return",
        ).reorder_levels(["instrument", "datetime"]).sort_index()

        ic, report, risk = evaluate_adaptive_history(scores, returns, topk=2)

        self.assertEqual(len(ic), len(dates))
        self.assertEqual(len(report), len(dates))
        self.assertGreater(ic.mean(), 0.9)
        self.assertEqual(risk.index.tolist(), ["Sharpe", "Sortino", "Calmar"])


if __name__ == "__main__":
    unittest.main()
