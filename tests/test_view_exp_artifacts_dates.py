from datetime import date
from pathlib import Path
import unittest

from examples.view_exp_artifacts import default_cpo_provider_uri


class TestViewExpArtifactsDates(unittest.TestCase):
    def test_default_cpo_provider_uri_uses_current_day_on_weekday(self):
        self.assertEqual(
            default_cpo_provider_uri(date(2026, 5, 15)),
            Path.home() / "qlib" / "examples" / "data" / "20260515" / "qlib_data",
        )

    def test_default_cpo_provider_uri_uses_friday_data_on_weekend(self):
        expected = Path.home() / "qlib" / "examples" / "data" / "20260515" / "qlib_data"

        self.assertEqual(default_cpo_provider_uri(date(2026, 5, 16)), expected)
        self.assertEqual(default_cpo_provider_uri(date(2026, 5, 17)), expected)


if __name__ == "__main__":
    unittest.main()
