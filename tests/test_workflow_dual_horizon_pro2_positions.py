import tempfile
import unittest
from pathlib import Path

import pandas as pd

from examples.workflow_dual_horizon_pro2 import (
    ADAPTIVE_STRATEGY_NAME,
    INITIAL_CAPITAL,
    POSITION_COLUMNS,
    combine_adaptive_scores,
    format_strategy_console_summary,
    load_current_positions,
    persist_position_state,
    update_adaptive_positions,
    update_fixed_horizon_positions,
)


class TestWorkflowDualHorizonPro2Positions(unittest.TestCase):
    def setUp(self):
        self.empty_positions = pd.DataFrame(columns=POSITION_COLUMNS)
        self.scores = pd.Series({"AAA": 0.95, "BBB": 0.85, "CCC": 0.75, "DDD": 0.65})

    def test_fixed_horizon_records_positions_and_increments_holding_days(self):
        first_positions, first_actions = update_fixed_horizon_positions(
            self.empty_positions,
            self.scores,
            "2026-08-03",
            strategy="5D_Mid_Term",
            holding_period=5,
            topk=2,
        )
        next_positions, next_actions = update_fixed_horizon_positions(
            first_positions,
            self.scores,
            "2026-08-04",
            strategy="5D_Mid_Term",
            holding_period=5,
            topk=2,
        )

        self.assertEqual(first_positions["holding_days"].tolist(), [1, 1])
        self.assertEqual(first_positions["amount"].tolist(), [INITIAL_CAPITAL / 2] * 2)
        self.assertEqual(set(first_actions["action"]), {"BUY"})
        self.assertEqual(next_positions["holding_days"].tolist(), [2, 2])
        self.assertEqual(set(next_actions["action"]), {"HOLD"})

    def test_one_day_horizon_sells_on_next_trade_day(self):
        first_positions, _ = update_fixed_horizon_positions(
            self.empty_positions,
            self.scores,
            "2026-08-03",
            strategy="1D_Short_Term",
            holding_period=1,
            topk=2,
        )
        next_positions, next_actions = update_fixed_horizon_positions(
            first_positions,
            self.scores,
            "2026-08-04",
            strategy="1D_Short_Term",
            holding_period=1,
            topk=2,
        )

        self.assertEqual((next_actions["action"] == "SELL").sum(), 2)
        self.assertEqual((next_actions["action"] == "BUY").sum(), 2)
        self.assertTrue(set(first_positions["instrument"]).isdisjoint(next_positions["instrument"]))

    def test_adaptive_strategy_waits_for_minimum_hold_then_sells_weak_signal(self):
        first_positions, _ = update_adaptive_positions(
            self.empty_positions, self.scores, "2026-08-03", topk=2
        )
        weaker_scores = pd.Series({"CCC": 0.95, "DDD": 0.85, "AAA": 0.50, "BBB": 0.45})

        second_positions, second_actions = update_adaptive_positions(
            first_positions, weaker_scores, "2026-08-04", topk=2
        )

        sold = second_actions.loc[second_actions["action"] == "SELL"]
        self.assertEqual(set(sold["instrument"]), {"AAA", "BBB"})
        self.assertTrue((sold["holding_days"] == 2).all())
        self.assertEqual(set(second_positions["instrument"]), {"CCC", "DDD"})

    def test_same_day_rerun_does_not_increment_holding_days_or_duplicate_history(self):
        positions, actions = update_adaptive_positions(
            self.empty_positions, self.scores, "2026-08-03", topk=2
        )
        rerun_positions, rerun_actions = update_adaptive_positions(
            positions, self.scores, "2026-08-03", topk=2
        )

        self.assertEqual(rerun_positions["holding_days"].tolist(), [1, 1])
        with tempfile.TemporaryDirectory() as temp_dir:
            state_dir = Path(temp_dir)
            persist_position_state(ADAPTIVE_STRATEGY_NAME, positions, actions, state_dir)
            persist_position_state(ADAPTIVE_STRATEGY_NAME, rerun_positions, rerun_actions, state_dir)
            history = pd.read_pickle(state_dir / f"{ADAPTIVE_STRATEGY_NAME}_history.pkl")

        self.assertEqual(len(history), len(actions))
        self.assertEqual(set(history["action"]), {"BUY"})

    def test_combined_score_uses_forty_sixty_weights(self):
        combined = combine_adaptive_scores(
            {
                "1D_Short_Term": pd.Series({"AAA": 1.0, "BBB": 0.0}),
                "5D_Mid_Term": pd.Series({"AAA": 0.0, "BBB": 1.0}),
            }
        )

        self.assertAlmostEqual(combined["AAA"], 0.4)
        self.assertAlmostEqual(combined["BBB"], 0.6)
        self.assertEqual(combined.index[0], "BBB")

    def test_console_summary_prints_positions_and_five_metrics_for_each_strategy(self):
        positions, _ = update_adaptive_positions(
            self.empty_positions, self.scores, "2026-08-04", topk=2
        )
        history = pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp("2026-08-04"),
                    "strategy": "Adaptive_Exit",
                    "IC": 0.03,
                    "ICIR": 0.6,
                    "Sharpe": 1.2,
                    "Sortino": 1.4,
                    "Calmar": 0.8,
                }
            ]
        )

        summary = format_strategy_console_summary(
            {"Adaptive_Exit": positions}, history
        )

        for strategy in ("1D_Short_Term", "5D_Mid_Term", "Adaptive_Exit"):
            self.assertIn(strategy, summary)
        for column in ("instrument", "entry_date", "holding_days", "score", "rank", "weight", "amount"):
            self.assertIn(column, summary)
        for metric in ("IC", "ICIR", "Sharpe", "Sortino", "Calmar"):
            self.assertIn(metric, summary)
        self.assertIn("AAA", summary)

    def test_old_position_state_derives_amount_from_weight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_dir = Path(temp_dir)
            legacy = pd.DataFrame(
                [
                    {
                        "strategy": "1D_Short_Term",
                        "instrument": "AAA",
                        "entry_date": pd.Timestamp("2026-08-03"),
                        "last_updated": pd.Timestamp("2026-08-04"),
                        "holding_days": 2,
                        "score": 0.9,
                        "rank": 1,
                        "weight": 0.2,
                    }
                ]
            )
            legacy.to_pickle(state_dir / "1D_Short_Term_current.pkl")

            positions = load_current_positions("1D_Short_Term", state_dir)

        self.assertEqual(positions.loc[0, "amount"], 60000)


if __name__ == "__main__":
    unittest.main()
