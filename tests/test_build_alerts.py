import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_processing import build_alerts


def test_build_alerts_emits_gross_margin_alert_for_zero_margin():
    monthly_summary = pd.DataFrame([{"sales_amount": 1000}])
    kpi_summary = {"gross_margin_rate": 0.0}
    cashflow_forecast = pd.DataFrame({"cash_balance": [1000]})

    alerts = build_alerts(monthly_summary, kpi_summary, cashflow_forecast)

    assert any("粗利率" in alert for alert in alerts)


def test_build_alerts_returns_structured_sales_drop_payload():
    monthly_summary = pd.DataFrame(
        [
            {"sales_amount": 1_000},
            {"sales_amount": 600},
        ]
    )
    kpi_summary = {"gross_margin_rate": 0.72, "churn_rate": 0.02}
    cashflow_forecast = pd.DataFrame({"cash_balance": [1_000, 1_500]})

    alerts = build_alerts(monthly_summary, kpi_summary, cashflow_forecast)

    assert alerts, "Expected at least one alert for the steep sales drop"
    sales_alert = next(alert for alert in alerts if alert.get("metric") == "sales_amount")

    expected_keys = {
        "message",
        "cause",
        "condition",
        "action_label",
        "target_section",
        "severity",
        "metric",
        "metric_value",
        "doc_path",
    }
    missing_keys = expected_keys - sales_alert.keys()
    assert not missing_keys, f"Missing keys from sales alert payload: {missing_keys}"
    assert sales_alert["severity"] == "warning"
    assert sales_alert["metric_value"] == pytest.approx(600)


def test_build_alerts_marks_cash_shortfall_as_error():
    monthly_summary = pd.DataFrame(
        [
            {"sales_amount": 1_000},
            {"sales_amount": 1_000},
        ]
    )
    kpi_summary = {"gross_margin_rate": 0.65, "churn_rate": 0.03}
    cashflow_forecast = pd.DataFrame({"cash_balance": [500, -200]})

    alerts = build_alerts(monthly_summary, kpi_summary, cashflow_forecast)

    cash_alert = next(alert for alert in alerts if alert.get("metric") == "cash_balance")
    assert cash_alert["severity"] == "error"
    assert "資金残高" in cash_alert["message"]
