from datetime import date
import unittest

from examples.dual_horizon_dates import get_data_dates


class TestWorkflowDualHorizonDates(unittest.TestCase):
    def test_weekday_uses_current_day(self):
        self.assertEqual(get_data_dates(date(2026, 5, 15)), ("20260515", "20260514"))

    def test_saturday_uses_previous_friday(self):
        self.assertEqual(get_data_dates(date(2026, 5, 16)), ("20260515", "20260514"))

    def test_sunday_uses_previous_friday(self):
        self.assertEqual(get_data_dates(date(2026, 5, 17)), ("20260515", "20260514"))

    def test_monday_uses_current_day(self):
        self.assertEqual(get_data_dates(date(2026, 5, 18)), ("20260518", "20260517"))


if __name__ == "__main__":
    unittest.main()
