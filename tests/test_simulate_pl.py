"""Tests for the PL simulation helpers."""

from __future__ import annotations

import unittest

from data_processing import simulate_pl


class SimulatePLTestCase(unittest.TestCase):
    """Regression tests for :func:`simulate_pl`."""

    def test_cost_ratio_is_capped_below_one(self) -> None:
        """Even at maximum adjustment the gross profit should stay non-negative."""

        base_pl = {
            "sales": 1_000.0,
            "cogs": 950.0,
            "sga": 400.0,
            "gross_profit": 50.0,
        }

        result = simulate_pl(
            base_pl,
            sales_growth_rate=0.0,
            cost_rate_adjustment=0.1,
            sga_change_rate=0.0,
            additional_ad_cost=0.0,
        )

        scenario_sales = result.loc[result["項目"] == "売上高", "シナリオ"].iloc[0]
        scenario_cogs = result.loc[result["項目"] == "売上原価", "シナリオ"].iloc[0]
        scenario_gross = result.loc[result["項目"] == "粗利", "シナリオ"].iloc[0]

        self.assertGreaterEqual(scenario_gross, 0.0)
        scenario_cost_ratio = scenario_cogs / scenario_sales if scenario_sales else 0.0
        self.assertLessEqual(scenario_cost_ratio, 0.99)


if __name__ == "__main__":
    unittest.main()
