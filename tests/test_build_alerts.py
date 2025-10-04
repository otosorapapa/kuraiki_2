import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_processing import build_alerts


def test_build_alerts_emits_gross_margin_alert_for_zero_margin():
    monthly_summary = pd.DataFrame([{"sales_amount": 1000}])
    kpi_summary = {"gross_margin_rate": 0.0}
    cashflow_forecast = pd.DataFrame({"cash_balance": [1000]})

    alerts = build_alerts(monthly_summary, kpi_summary, cashflow_forecast)

    assert any("粗利率" in alert for alert in alerts)
