"""Utility functions for the くらしいきいき社向け計数管理Webアプリ."""
from __future__ import annotations

# TODO: pandasとnumpyを使ってデータ集計を行う
import io
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from pandas.api.types import is_period_dtype

try:  # pragma: no cover - optional dependency
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:  # pragma: no cover - statsmodels may be unavailable during tests
    SARIMAX = None

try:  # pragma: no cover - optional dependency
    import tensorflow as tf
except Exception:  # pragma: no cover - tensorflow may not be installed in some environments
    tf = None

# 共通で利用する列名の定義
NORMALIZED_SALES_COLUMNS = [
    "order_date",
    "channel",
    "store",
    "product_code",
    "product_name",
    "category",
    "quantity",
    "sales_amount",
    "customer_id",
    "campaign",
]

SALES_COLUMN_ALIASES: Dict[str, List[str]] = {
    "order_date": ["注文日", "注文日時", "Date", "order_date", "注文年月日"],
    "channel": ["チャネル", "販売チャネル", "channel", "モール"],
    "store": ["店舗", "店名", "支店", "store", "倉庫"],
    "product_code": ["商品コード", "SKU", "品番", "product_code", "商品番号"],
    "product_name": ["商品名", "品名", "product", "product_name"],
    "category": ["カテゴリ", "カテゴリー", "category", "商品カテゴリ"],
    "quantity": ["数量", "個数", "quantity", "qty"],
    "sales_amount": ["売上", "売上高", "金額", "sales", "sales_amount", "合計金額"],
    "customer_id": ["顧客ID", "customer_id", "会員ID", "購入者ID"],
    "campaign": ["キャンペーン", "広告施策", "campaign", "施策名"],
}

COST_COLUMN_ALIASES: Dict[str, List[str]] = {
    "product_code": ["商品コード", "product_code", "SKU", "品番"],
    "product_name": ["商品名", "品名", "product_name", "品目名"],
    "category": ["カテゴリ", "カテゴリー", "category"],
    "price": ["売価", "単価", "price", "販売価格"],
    "cost": ["原価", "仕入原価", "cost"],
    "cost_rate": ["原価率", "cost_rate"],
}

SUBSCRIPTION_COLUMN_ALIASES: Dict[str, List[str]] = {
    "month": ["month", "年月", "月", "date", "対象月"],
    "active_customers": ["active_customers", "アクティブ顧客数", "継続会員数", "契約数", "有効会員数"],
    "new_customers": ["new_customers", "新規顧客数", "新規獲得数"],
    "repeat_customers": ["repeat_customers", "リピート顧客数", "継続購入者数"],
    "cancelled_subscriptions": ["cancelled_subscriptions", "解約件数", "解約数", "キャンセル数"],
    "previous_active_customers": ["previous_active_customers", "前月契約数", "前月アクティブ顧客"],
    "marketing_cost": ["marketing_cost", "広告費", "販促費", "marketing"],
    "ltv": ["ltv", "LTV", "顧客生涯価値"],
    "total_sales": ["total_sales", "売上高", "sales", "売上"],
}

DEFAULT_CHANNEL_FEE_RATES: Dict[str, float] = {
    "自社サイト": 0.03,
    "楽天市場": 0.12,
    "Amazon": 0.15,
    "Yahoo!ショッピング": 0.10,
}

DEFAULT_FIXED_COST = 2_500_000  # 人件費や管理費などの固定費（目安）
DEFAULT_LOAN_REPAYMENT = 600_000  # 月次の借入返済額の仮値
DEFAULT_DORMANCY_DAYS = 120  # 休眠判定に用いる前回購入からの日数


@dataclass
class ValidationMessage:
    """Represents a validation result item for loaded datasets."""

    level: str
    message: str
    count: Optional[int] = None
    sample: Optional[pd.DataFrame] = None


@dataclass
class ValidationReport:
    """Aggregated validation results for sales data ingestion."""

    messages: List[ValidationMessage] = field(default_factory=list)
    duplicate_rows: pd.DataFrame = field(default_factory=pd.DataFrame)

    def add_message(
        self,
        level: str,
        message: str,
        *,
        count: Optional[int] = None,
        sample: Optional[pd.DataFrame] = None,
    ) -> None:
        """Append a validation message, storing a small sample if provided."""

        sample_to_store: Optional[pd.DataFrame] = None
        if sample is not None and not sample.empty:
            sample_to_store = sample.copy().head(50)
        self.messages.append(
            ValidationMessage(level=level, message=message, count=count, sample=sample_to_store)
        )

    def add_duplicates(self, duplicates: pd.DataFrame) -> None:
        """Store duplicate records detected during validation."""

        if duplicates is None or duplicates.empty:
            return
        if self.duplicate_rows.empty:
            self.duplicate_rows = duplicates.copy()
        else:
            self.duplicate_rows = (
                pd.concat([self.duplicate_rows, duplicates], ignore_index=True)
                .drop_duplicates()
                .reset_index(drop=True)
            )

    def extend(self, other: Optional["ValidationReport"]) -> None:
        """Merge another validation report into this one."""

        if other is None:
            return
        self.messages.extend(other.messages)
        self.add_duplicates(other.duplicate_rows)

    def has_errors(self) -> bool:
        return any(msg.level == "error" for msg in self.messages)

    def has_warnings(self) -> bool:
        return any(msg.level == "warning" for msg in self.messages)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return bool(self.messages) or not self.duplicate_rows.empty


@dataclass
class ForecastModelArtifact:
    """Container for trained売上予測モデル."""

    model_type: str
    freq: str
    training_series: pd.Series
    fitted_model: Any
    window_size: int = 12
    scaler_min: Optional[float] = None
    scaler_scale: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)

def detect_duplicate_rows(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return rows that are duplicated for the given subset of columns."""

    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else [])

    dedupe_subset = subset or [
        "order_date",
        "channel",
        "product_code",
        "customer_id",
        "sales_amount",
    ]
    duplicates = df[df.duplicated(subset=dedupe_subset, keep=False)].copy()
    if duplicates.empty:
        return pd.DataFrame(columns=df.columns)
    return duplicates.sort_values(by=dedupe_subset).reset_index(drop=True)


def detect_channel_from_filename(filename: Optional[str]) -> Optional[str]:
    """ファイル名からチャネル名を推定する。"""
    if not filename:
        return None
    name = filename.lower()
    if "rakuten" in name or "楽天" in name:
        return "楽天市場"
    if "amazon" in name:
        return "Amazon"
    if "yahoo" in name:
        return "Yahoo!ショッピング"
    if "shop" in name or "ec" in name or "自社" in name:
        return "自社サイト"
    return None


def _build_rename_map(columns: Iterable[str], alias_config: Dict[str, List[str]]) -> Dict[str, str]:
    """指定した列から正規化用のrename辞書を作成する。"""
    rename_map: Dict[str, str] = {}
    normalized_cols = [col.lower() for col in columns]
    for canonical, aliases in alias_config.items():
        for alias in aliases:
            if alias.lower() in normalized_cols:
                idx = normalized_cols.index(alias.lower())
                rename_map[list(columns)[idx]] = canonical
                break
    return rename_map


def normalize_sales_df(df: pd.DataFrame, channel: Optional[str] = None) -> pd.DataFrame:
    """売上データを統一フォーマットに整形する。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=NORMALIZED_SALES_COLUMNS)

    rename_map = _build_rename_map(df.columns, SALES_COLUMN_ALIASES)
    normalized = df.rename(columns=rename_map).copy()

    for column in NORMALIZED_SALES_COLUMNS:
        if column not in normalized.columns:
            if column == "channel" and channel:
                normalized[column] = channel
            elif column == "quantity":
                normalized[column] = 1
            else:
                normalized[column] = np.nan

    normalized = normalized[NORMALIZED_SALES_COLUMNS]
    normalized["order_date"] = pd.to_datetime(normalized["order_date"], errors="coerce")
    normalized["channel"] = normalized["channel"].fillna(channel or "不明").astype(str)
    if "store" in normalized.columns:
        normalized["store"] = normalized["store"].fillna("全社").astype(str)
    normalized["product_code"] = normalized["product_code"].fillna("NA").astype(str)
    normalized["product_name"] = normalized["product_name"].fillna("不明商品").astype(str)
    normalized["category"] = normalized["category"].fillna("未分類").astype(str)
    normalized["quantity"] = pd.to_numeric(normalized["quantity"], errors="coerce").fillna(1)
    normalized["sales_amount"] = pd.to_numeric(normalized["sales_amount"], errors="coerce").fillna(0.0)
    normalized["customer_id"] = normalized["customer_id"].fillna("anonymous").astype(str)
    if "campaign" in normalized.columns:
        normalized["campaign"] = normalized["campaign"].fillna("キャンペーン未設定").astype(str)
    normalized = normalized.dropna(subset=["order_date"])

    # 単価を推計しておく（quantityが0の場合は回避）
    normalized["unit_price"] = normalized.apply(
        lambda row: row["sales_amount"] / row["quantity"] if row["quantity"] else row["sales_amount"],
        axis=1,
    )
    normalized["order_month"] = normalized["order_date"].dt.to_period("M")
    return normalized


def validate_sales_integrity(
    raw_df: pd.DataFrame,
    normalized_df: pd.DataFrame,
    *,
    source: Optional[str] = None,
) -> ValidationReport:
    """Validate essential columns and value ranges for sales data."""

    report = ValidationReport()
    label = f"[{source}] " if source else ""

    if raw_df is None:
        report.add_message("error", f"{label}ファイルが読み込めませんでした。")
        return report

    if raw_df.empty:
        report.add_message("warning", f"{label}データ行が存在しません。空のファイルが指定されている可能性があります。")
        return report

    rename_map = _build_rename_map(raw_df.columns, SALES_COLUMN_ALIASES)
    resolved_columns = set(rename_map.values())

    essential_columns = {
        "order_date": "注文日",
        "sales_amount": "売上金額",
        "quantity": "数量",
    }
    for canonical, display in essential_columns.items():
        has_column = canonical in resolved_columns or canonical in raw_df.columns
        if not has_column:
            report.add_message(
                "error",
                f"{label}必須項目「{display}」が見つかりません。列名が正しくマッピングされているか確認してください。",
            )

    if normalized_df is None or normalized_df.empty:
        report.add_message(
            "warning",
            f"{label}正しく読み込めた売上データがありませんでした。ファイル内容を確認してください。",
        )
        return report

    date_col = next((col for col, canonical in rename_map.items() if canonical == "order_date"), None)
    if date_col:
        parsed_dates = pd.to_datetime(raw_df[date_col], errors="coerce")
        invalid_dates = raw_df.loc[parsed_dates.isna()]
        if not invalid_dates.empty:
            report.add_message(
                "warning",
                f"{label}日付が解釈できなかったレコードが{len(invalid_dates)}件あり、読み込み対象から除外しました。",
                count=int(invalid_dates.shape[0]),
                sample=invalid_dates[[date_col]],
            )

    quantity_col = next((col for col, canonical in rename_map.items() if canonical == "quantity"), None)
    if quantity_col:
        raw_quantity = pd.to_numeric(raw_df[quantity_col], errors="coerce")
        invalid_quantity = raw_df.loc[raw_quantity.isna() | (raw_quantity <= 0)]
        if not invalid_quantity.empty:
            report.add_message(
                "error",
                f"{label}数量が0以下、または未入力のレコードが{len(invalid_quantity)}件あります。",
                count=int(invalid_quantity.shape[0]),
                sample=invalid_quantity[[quantity_col]],
            )

    sales_col = next((col for col, canonical in rename_map.items() if canonical == "sales_amount"), None)
    if sales_col:
        raw_sales = pd.to_numeric(raw_df[sales_col], errors="coerce")
        missing_sales = raw_df.loc[raw_sales.isna()]
        if not missing_sales.empty:
            report.add_message(
                "error",
                f"{label}売上高が数値として読み取れないレコードが{len(missing_sales)}件あります。",
                count=int(missing_sales.shape[0]),
                sample=missing_sales[[sales_col]],
            )
        negative_sales = raw_df.loc[raw_sales < 0]
        if not negative_sales.empty:
            report.add_message(
                "error",
                f"{label}売上高がマイナスのレコードが{len(negative_sales)}件あります。",
                count=int(negative_sales.shape[0]),
                sample=negative_sales[[sales_col]],
            )
        zero_sales = raw_df.loc[raw_sales == 0]
        if not zero_sales.empty:
            report.add_message(
                "warning",
                f"{label}売上高が0円のレコードが{len(zero_sales)}件あります。無償提供等で問題ないか確認してください。",
                count=int(zero_sales.shape[0]),
                sample=zero_sales[[sales_col]],
            )

    invalid_unit_price = normalized_df[
        (~np.isfinite(normalized_df["unit_price"])) | (normalized_df["unit_price"] <= 0)
    ]
    if not invalid_unit_price.empty:
        report.add_message(
            "error",
            f"{label}単価が0以下、または計算できないレコードが{len(invalid_unit_price)}件あります。",
            count=int(invalid_unit_price.shape[0]),
            sample=invalid_unit_price[
                ["order_date", "channel", "product_code", "quantity", "sales_amount", "unit_price"]
            ],
        )

    extreme_unit_price = normalized_df[normalized_df["unit_price"] > 1_000_000]
    if not extreme_unit_price.empty:
        report.add_message(
            "warning",
            f"{label}単価が1,000,000円を超えるレコードが{len(extreme_unit_price)}件あります。異常値でないか確認してください。",
            count=int(extreme_unit_price.shape[0]),
            sample=extreme_unit_price[
                ["order_date", "channel", "product_code", "quantity", "sales_amount", "unit_price"]
            ],
        )

    duplicates = detect_duplicate_rows(normalized_df)
    if not duplicates.empty:
        report.add_message(
            "warning",
            f"{label}重複している可能性のあるレコードが{len(duplicates)}件あります。",
            count=int(duplicates.shape[0]),
        )
        report.add_duplicates(duplicates)

    return report


def load_sales_workbook(
    uploaded_file,
    channel_hint: Optional[str] = None,
) -> Tuple[pd.DataFrame, ValidationReport]:
    """アップロードされたExcel/CSVを読み込み正規化する。"""

    if uploaded_file is None:
        return pd.DataFrame(columns=NORMALIZED_SALES_COLUMNS), ValidationReport()

    try:
        uploaded_file.seek(0)
    except Exception:  # pragma: no cover - Streamlit's UploadedFile may not support seek
        pass

    read_errors: List[str] = []

    try:
        df = pd.read_excel(uploaded_file)
    except ValueError as exc:
        read_errors.append(str(exc))
        try:
            uploaded_file.seek(0)
        except Exception:  # pragma: no cover - as above
            pass
        try:
            df = pd.read_csv(uploaded_file)
        except pd.errors.EmptyDataError as exc:
            read_errors.append(str(exc))
            df = pd.DataFrame()
        except Exception as exc:  # pragma: no cover - unexpected parsers
            read_errors.append(str(exc))
            df = pd.DataFrame()
    except Exception as exc:  # pragma: no cover - general fallback
        read_errors.append(str(exc))
        df = pd.DataFrame()

    detected = channel_hint or detect_channel_from_filename(getattr(uploaded_file, "name", None))
    normalized = normalize_sales_df(df, channel=detected)

    source_name_parts: List[str] = []
    if channel_hint:
        source_name_parts.append(channel_hint)
    elif detected:
        source_name_parts.append(detected)
    file_name = getattr(uploaded_file, "name", None)
    if file_name:
        source_name_parts.append(file_name)
    source_name = " - ".join(source_name_parts) if source_name_parts else file_name

    validation = validate_sales_integrity(df, normalized, source=source_name)
    if read_errors and normalized.empty:
        validation.add_message(
            "error",
            f"{source_name or 'アップロードデータ'}の読み込みに失敗しました: {read_errors[-1]}",
        )
    return normalized, validation


def load_sales_files(files_by_channel: Dict[str, List]) -> Tuple[pd.DataFrame, ValidationReport]:
    """チャネルごとのファイル群を統合した売上データを作成する。"""

    frames: List[pd.DataFrame] = []
    combined_report = ValidationReport()

    for channel, files in files_by_channel.items():
        if not files:
            continue
        for uploaded in files:
            normalized, report = load_sales_workbook(uploaded, channel_hint=channel)
            combined_report.extend(report)
            if not normalized.empty:
                frames.append(normalized)

    if not frames:
        return pd.DataFrame(columns=NORMALIZED_SALES_COLUMNS), combined_report

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("order_date", inplace=True)

    duplicates = detect_duplicate_rows(combined)
    if not duplicates.empty:
        previous_count = len(combined_report.duplicate_rows)
        combined_report.add_duplicates(duplicates)
        if len(combined_report.duplicate_rows) > previous_count:
            combined_report.add_message(
                "warning",
                f"取り込んだ売上データ全体で重複しているレコードが{len(duplicates)}件見つかりました。",
                count=int(duplicates.shape[0]),
            )

    return combined, combined_report


def fetch_sales_from_endpoint(
    endpoint: str,
    *,
    token: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    channel_hint: Optional[str] = None,
    timeout: int = 30,
) -> Tuple[pd.DataFrame, ValidationReport]:
    """Fetch sales data from a remote API endpoint and normalize it."""

    report = ValidationReport()
    if not endpoint:
        report.add_message("error", "APIエンドポイントが設定されていません。")
        return pd.DataFrame(columns=NORMALIZED_SALES_COLUMNS), report

    headers = {"Accept": "*/*"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        report.add_message("error", f"APIからのデータ取得に失敗しました: {exc}")
        return pd.DataFrame(columns=NORMALIZED_SALES_COLUMNS), report

    content_type = (response.headers.get("Content-Type") or "").lower()

    try:
        if "json" in content_type or endpoint.lower().endswith(".json"):
            payload = response.json()
            if isinstance(payload, dict):
                for key in ("data", "results", "items", "orders", "records"):
                    value = payload.get(key)
                    if isinstance(value, list):
                        payload = value
                        break
                else:
                    payload = [payload]
            if isinstance(payload, list):
                df = pd.DataFrame(payload)
            else:
                df = pd.DataFrame(payload)
        elif any(endpoint.lower().endswith(ext) for ext in (".xls", ".xlsx")) or "excel" in content_type:
            df = pd.read_excel(io.BytesIO(response.content))
        else:
            df = pd.read_csv(io.StringIO(response.text))
    except ValueError as exc:
        report.add_message("error", f"APIレスポンスの解析に失敗しました: {exc}")
        return pd.DataFrame(columns=NORMALIZED_SALES_COLUMNS), report
    except pd.errors.EmptyDataError:
        report.add_message("warning", "APIレスポンスにデータが含まれていませんでした。")
        return pd.DataFrame(columns=NORMALIZED_SALES_COLUMNS), report

    normalized = normalize_sales_df(df, channel=channel_hint)
    source = channel_hint or endpoint
    validation = validate_sales_integrity(df, normalized, source=source)
    if normalized.empty:
        validation.add_message("warning", f"{source} から取得したデータは空でした。")

    return normalized, validation


def load_cost_workbook(uploaded_file) -> pd.DataFrame:
    """原価率表を読み込む。"""
    if uploaded_file is None:
        return pd.DataFrame(columns=["product_code", "product_name", "category", "price", "cost", "cost_rate"])

    try:
        df = pd.read_excel(uploaded_file)
    except ValueError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

    rename_map = _build_rename_map(df.columns, COST_COLUMN_ALIASES)
    normalized = df.rename(columns=rename_map)

    for column in ["product_code", "product_name", "category", "price", "cost", "cost_rate"]:
        if column not in normalized.columns:
            normalized[column] = np.nan

    normalized["product_code"] = normalized["product_code"].fillna("NA").astype(str)
    normalized["product_name"] = normalized["product_name"].fillna("不明商品").astype(str)
    normalized["category"] = normalized["category"].fillna("未分類").astype(str)
    normalized["price"] = pd.to_numeric(normalized["price"], errors="coerce")
    normalized["cost"] = pd.to_numeric(normalized["cost"], errors="coerce")
    normalized["cost_rate"] = pd.to_numeric(normalized["cost_rate"], errors="coerce")

    if normalized["cost_rate"].isna().any():
        normalized.loc[:, "cost_rate"] = normalized.apply(
            lambda row: row["cost"] / row["price"] if row["price"] else np.nan,
            axis=1,
        )
    normalized["gross_margin_rate"] = 1 - normalized["cost_rate"].fillna(0)
    return normalized


def load_subscription_workbook(uploaded_file) -> pd.DataFrame:
    """サブスク/KPIデータを読み込む。"""
    if uploaded_file is None:
        return pd.DataFrame(
            columns=[
                "month",
                "active_customers",
                "previous_active_customers",
                "new_customers",
                "repeat_customers",
                "cancelled_subscriptions",
                "marketing_cost",
                "ltv",
                "total_sales",
            ]
        )

    try:
        df = pd.read_excel(uploaded_file)
    except ValueError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

    rename_map = _build_rename_map(df.columns, SUBSCRIPTION_COLUMN_ALIASES)
    normalized = df.rename(columns=rename_map).copy()

    for column in [
        "month",
        "active_customers",
        "previous_active_customers",
        "new_customers",
        "repeat_customers",
        "cancelled_subscriptions",
        "marketing_cost",
        "ltv",
        "total_sales",
    ]:
        if column not in normalized.columns:
            normalized[column] = np.nan

    normalized["month"] = pd.to_datetime(normalized["month"], errors="coerce").dt.to_period("M")
    numeric_cols = [
        "active_customers",
        "previous_active_customers",
        "new_customers",
        "repeat_customers",
        "cancelled_subscriptions",
        "marketing_cost",
        "ltv",
        "total_sales",
    ]
    for col in numeric_cols:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    return normalized


def merge_sales_and_costs(sales_df: pd.DataFrame, cost_df: pd.DataFrame) -> pd.DataFrame:
    """売上データに原価情報を結合し、粗利を計算する。"""
    if sales_df.empty:
        return sales_df

    merged = sales_df.copy()
    if not cost_df.empty:
        join_keys = ["product_code"] if "product_code" in cost_df.columns else []
        if not join_keys or cost_df["product_code"].eq("NA").all():
            join_keys = ["product_name"]
        merged = merged.merge(cost_df, on=join_keys, how="left", suffixes=("", "_cost"))
    else:
        merged = merged.assign(price=np.nan, cost=np.nan, cost_rate=np.nan, gross_margin_rate=np.nan)

    merged["cost_rate"] = merged["cost_rate"].fillna(0.3)
    merged["cost_rate"] = merged["cost_rate"].clip(0, 0.95)
    if "gross_margin_rate" in merged.columns:
        merged["gross_margin_rate"] = merged["gross_margin_rate"].fillna(1 - merged["cost_rate"])
    else:
        merged["gross_margin_rate"] = 1 - merged["cost_rate"]
    merged["estimated_cost"] = merged["sales_amount"] * merged["cost_rate"]
    merged["gross_profit"] = merged["sales_amount"] - merged["estimated_cost"]
    merged["channel_fee"] = merged["channel"].map(DEFAULT_CHANNEL_FEE_RATES).fillna(0)
    merged["channel_fee_amount"] = merged["sales_amount"] * merged["channel_fee"]
    merged["net_gross_profit"] = merged["gross_profit"] - merged["channel_fee_amount"]
    return merged


def validate_channel_fees(merged_df: pd.DataFrame) -> ValidationReport:
    """Validate that channel fee related fields are populated with sane values."""

    report = ValidationReport()
    if merged_df is None or merged_df.empty:
        return report

    known_channels = set(DEFAULT_CHANNEL_FEE_RATES.keys())
    present_channels = {ch for ch in merged_df["channel"].dropna().unique()}
    unknown_channels = sorted(present_channels - known_channels)
    if unknown_channels:
        report.add_message(
            "warning",
            "チャネル手数料率が未設定のチャネルがあります: " + ", ".join(unknown_channels) + "。DEFAULT_CHANNEL_FEE_RATESに追加してください。",
        )

    if "channel_fee_amount" in merged_df.columns:
        negative_fee = merged_df[merged_df["channel_fee_amount"] < 0]
        if not negative_fee.empty:
            report.add_message(
                "error",
                f"手数料金額がマイナスになっているレコードが{len(negative_fee)}件あります。",
                count=int(negative_fee.shape[0]),
                sample=negative_fee[["order_date", "channel", "sales_amount", "channel_fee_amount"]],
            )

        excessive_fee = merged_df[
            (merged_df["sales_amount"].abs() > 0)
            & (merged_df["channel_fee_amount"].abs() > merged_df["sales_amount"].abs())
        ]
        if not excessive_fee.empty:
            report.add_message(
                "warning",
                f"手数料金額が売上高を上回っているレコードが{len(excessive_fee)}件あります。設定ミスがないか確認してください。",
                count=int(excessive_fee.shape[0]),
                sample=excessive_fee[["order_date", "channel", "sales_amount", "channel_fee_amount"]],
            )

    return report


def aggregate_sales(df: pd.DataFrame, group_fields: List[str]) -> pd.DataFrame:
    """汎用的な売上集計処理。"""
    if df.empty:
        return pd.DataFrame(columns=group_fields + ["sales_amount"])
    aggregated = df.groupby(group_fields)["sales_amount"].sum().reset_index()
    return aggregated


def _mean_absolute_percentage_error(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Return the平均絶対パーセント誤差."""

    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    if actual_arr.size == 0 or predicted_arr.size == 0:
        return float("nan")
    mask = np.abs(actual_arr) > 1e-8
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((actual_arr[mask] - predicted_arr[mask]) / actual_arr[mask])))


def _compute_naive_mape(series: pd.Series) -> float:
    """Compute naiveな前月値比較のMAPE."""

    if series is None or series.empty:
        return float("nan")
    shifted = series.shift(1).dropna()
    if shifted.empty:
        return float("nan")
    aligned_actual = series.iloc[1 : len(series)]
    return _mean_absolute_percentage_error(aligned_actual.values, shifted.values)


def _z_score_from_confidence(confidence_level: float) -> float:
    """Approximate Z値 from信頼水準."""

    lookup = {
        0.80: 1.2816,
        0.85: 1.4395,
        0.90: 1.6449,
        0.95: 1.96,
        0.98: 2.3263,
        0.99: 2.5758,
    }
    rounded = round(float(confidence_level), 2)
    return float(lookup.get(rounded, lookup.get(0.95)))


def _prepare_monthly_sales_series(
    summary_df: pd.DataFrame,
    freq: str,
    *,
    min_periods: int = 6,
) -> pd.Series:
    """月次売上サマリからモデリング用の時系列を生成する。"""

    if summary_df is None or summary_df.empty:
        raise ValueError("月次売上サマリが空のため学習できません。")
    if "order_month" not in summary_df.columns or "sales_amount" not in summary_df.columns:
        raise ValueError("月次サマリに必要な列が不足しています。")

    working = summary_df[["order_month", "sales_amount"]].copy()
    working.sort_values("order_month", inplace=True)
    series = working.set_index("order_month")["sales_amount"].astype(float)

    if isinstance(series.index, pd.PeriodIndex):
        series.index = series.index.asfreq(freq)
    elif isinstance(series.index, pd.DatetimeIndex):
        series.index = series.index.to_period(freq)
    else:
        series.index = pd.PeriodIndex(series.index, freq=freq)

    full_index = pd.period_range(series.index.min(), series.index.max(), freq=freq)
    series = series.reindex(full_index)
    if series.isna().all():
        raise ValueError("売上時系列に有効な値がありません。")
    series = series.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    if series.count() < min_periods:
        raise ValueError("学習に十分な期間のデータがありません。")

    return series


def _build_lstm_training_data(
    series: pd.Series, window_size: int
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """LSTM用の教師データとスケーリング情報を返す。"""

    values = series.astype(float).values.astype("float32")
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    scale = float(max(max_val - min_val, 1.0))
    normalized = (values - min_val) / scale
    if len(normalized) <= window_size:
        raise ValueError("LSTM学習にはウィンドウサイズより多いサンプルが必要です。")
    sequences: List[np.ndarray] = []
    targets: List[float] = []
    for idx in range(window_size, len(normalized)):
        sequences.append(normalized[idx - window_size : idx])
        targets.append(normalized[idx])
    X = np.asarray(sequences, dtype="float32")
    y = np.asarray(targets, dtype="float32")
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, min_val, scale


def train_forecast_model(
    monthly_summary: pd.DataFrame,
    *,
    model_type: str = "arima",
    freq: str = "M",
    window_size: int = 12,
    arima_order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    lstm_params: Optional[Dict[str, Any]] = None,
    sarimax_kwargs: Optional[Dict[str, Any]] = None,
) -> ForecastModelArtifact:
    """月次売上データからARIMA/LSTMモデルを学習する。"""

    series = _prepare_monthly_sales_series(monthly_summary, freq, min_periods=max(6, window_size))
    metrics: Dict[str, float] = {"avg_sales": float(series.tail(min(len(series), 12)).mean())}
    model_type_normalized = (model_type or "arima").lower()

    if model_type_normalized == "lstm":
        if tf is None:  # pragma: no cover - depends on optional dependency
            raise ImportError("TensorFlowがインストールされていないためLSTMモデルを利用できません。")
        lstm_params = lstm_params or {}
        units = int(lstm_params.get("units", 32))
        dropout = float(lstm_params.get("dropout", 0.0))
        epochs = int(lstm_params.get("epochs", 120))
        batch_size = max(1, int(lstm_params.get("batch_size", 8)))

        X, y, min_val, scale = _build_lstm_training_data(series, window_size)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(window_size, 1)),
                tf.keras.layers.LSTM(units, dropout=dropout),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        if len(X) > 0:
            model.fit(X, y, epochs=max(1, epochs), batch_size=batch_size, verbose=0)
        predictions_scaled = model.predict(X, verbose=0).flatten()
        predictions = predictions_scaled * scale + min_val
        actual = series.iloc[window_size:].astype(float).values
        metrics["model_mape"] = _mean_absolute_percentage_error(actual, predictions)
        metrics["baseline_mape"] = _compute_naive_mape(series)
        if not math.isnan(metrics["baseline_mape"]) and not math.isnan(metrics["model_mape"]):
            metrics["mape_improvement"] = max(0.0, metrics["baseline_mape"] - metrics["model_mape"])
        else:
            metrics["mape_improvement"] = float("nan")
        metrics["residual_std"] = float(np.nanstd(actual - predictions))
        return ForecastModelArtifact(
            model_type="lstm",
            freq=freq,
            training_series=series,
            fitted_model=model,
            window_size=window_size,
            scaler_min=min_val,
            scaler_scale=scale,
            metrics=metrics,
        )

    if SARIMAX is None:  # pragma: no cover - depends on optional dependency
        raise ImportError("statsmodelsがインストールされていないためARIMAモデルを利用できません。")
    if len(series) < 3:
        raise ValueError("ARIMAモデルの学習には最低3期間のデータが必要です。")
    sarimax_kwargs = sarimax_kwargs or {}
    model = SARIMAX(
        series,
        order=arima_order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        **sarimax_kwargs,
    )
    results = model.fit(disp=False)
    fitted_values = results.fittedvalues.astype(float)
    aligned_pred = fitted_values.iloc[1:].astype(float)
    aligned_actual = series.iloc[1:].astype(float)
    aligned_pred = aligned_pred.reindex(aligned_actual.index)
    metrics["model_mape"] = _mean_absolute_percentage_error(aligned_actual.values, aligned_pred.values)
    metrics["baseline_mape"] = _compute_naive_mape(series)
    if not math.isnan(metrics["baseline_mape"]) and not math.isnan(metrics["model_mape"]):
        metrics["mape_improvement"] = max(0.0, metrics["baseline_mape"] - metrics["model_mape"])
    else:
        metrics["mape_improvement"] = float("nan")
    metrics["residual_std"] = float(np.nanstd(aligned_actual.values - aligned_pred.values))

    return ForecastModelArtifact(
        model_type="arima",
        freq=freq,
        training_series=series,
        fitted_model=results,
        window_size=window_size,
        metrics=metrics,
    )


def predict_sales_forecast(
    artifact: ForecastModelArtifact,
    periods: int = 6,
    *,
    confidence_level: float = 0.9,
) -> pd.DataFrame:
    """学習済みモデルから売上予測と信頼区間を生成する。"""

    if artifact is None:
        raise ValueError("予測モデルが未提供です。")
    if periods <= 0:
        raise ValueError("予測期間は1以上で指定してください。")

    freq = artifact.freq or "M"
    last_period = artifact.training_series.index[-1]
    start_period = last_period + 1
    forecast_index = pd.period_range(start_period, periods=periods, freq=freq)

    lower = np.full(periods, np.nan, dtype=float)
    upper = np.full(periods, np.nan, dtype=float)

    if artifact.model_type == "arima":
        forecast_result = artifact.fitted_model.get_forecast(steps=periods)
        predicted_mean = pd.Series(forecast_result.predicted_mean, index=forecast_index).astype(float)
        alpha = 1.0 - confidence_level
        conf_int = forecast_result.conf_int(alpha=alpha)
        if isinstance(conf_int, pd.DataFrame):
            lower = conf_int.iloc[:, 0].to_numpy(dtype=float)
            upper = conf_int.iloc[:, -1].to_numpy(dtype=float)
        else:
            lower = np.asarray(conf_int[:, 0], dtype=float)
            upper = np.asarray(conf_int[:, 1], dtype=float)
    elif artifact.model_type == "lstm":
        if tf is None:  # pragma: no cover - depends on optional dependency
            raise ImportError("TensorFlowが利用できません。")
        scale = artifact.scaler_scale or 1.0
        min_val = artifact.scaler_min or 0.0
        window_size = artifact.window_size
        history = artifact.training_series.astype(float).values.astype("float32")
        normalized_history = (history - min_val) / (scale if scale != 0 else 1.0)
        recent_window = normalized_history[-window_size:].copy()
        predictions: List[float] = []
        for step in range(periods):
            input_window = recent_window.reshape(1, window_size, 1)
            pred_scaled = artifact.fitted_model.predict(input_window, verbose=0)[0, 0]
            predictions.append(float(pred_scaled * scale + min_val))
            recent_window = np.append(recent_window[1:], pred_scaled)
        predicted_mean = pd.Series(predictions, index=forecast_index)
        residual_std = float(artifact.metrics.get("residual_std", 0.0))
        z_value = _z_score_from_confidence(confidence_level)
        lower = predicted_mean.to_numpy(dtype=float) - z_value * residual_std
        upper = predicted_mean.to_numpy(dtype=float) + z_value * residual_std
    else:
        raise ValueError(f"未対応のモデルタイプです: {artifact.model_type}")

    lower = np.maximum(0.0, lower)
    forecast_df = pd.DataFrame(
        {
            "month": forecast_index,
            "forecast": predicted_mean.to_numpy(dtype=float),
            "lower_ci": lower,
            "upper_ci": upper,
        }
    )
    forecast_df["confidence_level"] = confidence_level
    forecast_df["model_type"] = artifact.model_type
    return forecast_df


def apply_forecast_to_cashflow(
    plan_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    *,
    retention_rate: float = 0.75,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """売上予測をキャッシュフロー計画へ反映し、影響サマリを返す。"""

    if plan_df is None or plan_df.empty or forecast_df is None or forecast_df.empty:
        return (plan_df.copy() if isinstance(plan_df, pd.DataFrame) else pd.DataFrame(), pd.DataFrame())

    working = plan_df.copy()
    if not is_period_dtype(working["month"]):
        working["month"] = pd.PeriodIndex(working["month"], freq="M")

    forecast_working = forecast_df.copy()
    forecast_working["month"] = pd.PeriodIndex(forecast_working["month"], freq="M")
    forecast_working = forecast_working.rename(
        columns={
            "forecast": "sales_forecast",
            "lower_ci": "sales_forecast_lower",
            "upper_ci": "sales_forecast_upper",
        }
    )

    merged = working.merge(forecast_working, on="month", how="left")
    merged["baseline_operating_cf"] = merged["operating_cf"].astype(float)

    margin_ratio = 0.3
    if monthly_summary is not None and not monthly_summary.empty:
        total_sales = float(monthly_summary["sales_amount"].sum())
        total_net = float(monthly_summary["net_gross_profit"].sum())
        if total_sales > 0:
            margin_ratio = max(0.0, min(1.0, total_net / total_sales))

    for column in ["sales_forecast", "sales_forecast_lower", "sales_forecast_upper"]:
        if column in merged.columns:
            merged[column] = merged[column].astype(float)

    merged["operating_cf_forecast"] = merged["sales_forecast"] * margin_ratio * retention_rate
    merged["operating_cf_lower"] = merged["sales_forecast_lower"] * margin_ratio * retention_rate
    merged["operating_cf_upper"] = merged["sales_forecast_upper"] * margin_ratio * retention_rate

    mask = merged["operating_cf_forecast"].notna()
    merged.loc[mask, "operating_cf"] = merged.loc[mask, "operating_cf_forecast"]

    baseline_mean = float(merged["baseline_operating_cf"].mean())
    adjusted_mean = float(merged["operating_cf"].mean())
    baseline_total = float(merged["baseline_operating_cf"].sum())
    adjusted_total = float(merged["operating_cf"].sum())

    summary_df = pd.DataFrame(
        {
            "指標": ["平均営業CF", "累積営業CF"],
            "現状": [baseline_mean, baseline_total],
            "予測適用": [adjusted_mean, adjusted_total],
            "差分": [adjusted_mean - baseline_mean, adjusted_total - baseline_total],
        }
    )

    return merged, summary_df


def estimate_forecast_savings(
    artifact: Optional[ForecastModelArtifact],
    monthly_summary: Optional[pd.DataFrame],
    *,
    error_cost_factor: float = 0.12,
) -> pd.DataFrame:
    """予測誤差改善によるコスト削減効果を推計する。"""

    if artifact is None:
        return pd.DataFrame()

    metrics = artifact.metrics or {}
    baseline_mape = metrics.get("baseline_mape")
    model_mape = metrics.get("model_mape")
    improvement = metrics.get("mape_improvement")

    if baseline_mape is None or model_mape is None:
        return pd.DataFrame()

    avg_sales = float("nan")
    if monthly_summary is not None and not monthly_summary.empty:
        recent = monthly_summary.tail(12)
        if not recent.empty:
            avg_sales = float(recent["sales_amount"].mean())
    if not math.isfinite(avg_sales) or avg_sales <= 0:
        avg_sales = float(metrics.get("avg_sales", float("nan")))

    base_cost = avg_sales * error_cost_factor if math.isfinite(avg_sales) and avg_sales > 0 else float("nan")
    baseline_loss = base_cost * baseline_mape if math.isfinite(base_cost) else float("nan")
    improved_loss = base_cost * model_mape if math.isfinite(base_cost) else float("nan")
    annual_savings = (
        (baseline_loss - improved_loss) * 12 if math.isfinite(baseline_loss) and math.isfinite(improved_loss) else float("nan")
    )

    rows: List[Dict[str, float]] = []
    if math.isfinite(avg_sales):
        rows.append({"指標": "平均月次売上高", "現状": avg_sales, "予測適用": avg_sales, "差分": 0.0})
    if baseline_mape is not None and model_mape is not None:
        rows.append(
            {
                "指標": "予測MAPE",
                "現状": baseline_mape,
                "予測適用": model_mape,
                "差分": model_mape - baseline_mape,
            }
        )
    if improvement is not None and not math.isnan(improvement):
        rows.append({"指標": "誤差改善率", "現状": float("nan"), "予測適用": improvement, "差分": float("nan")})
    if math.isfinite(baseline_loss) and math.isfinite(improved_loss):
        rows.append(
            {
                "指標": "在庫・調達コスト(月次)",
                "現状": baseline_loss,
                "予測適用": improved_loss,
                "差分": improved_loss - baseline_loss,
            }
        )
    if math.isfinite(annual_savings):
        rows.append({"指標": "年間コスト削減見込", "現状": float("nan"), "予測適用": annual_savings, "差分": float("nan")})

    return pd.DataFrame(rows)


def monthly_sales_summary(df: pd.DataFrame) -> pd.DataFrame:
    """月次の売上と粗利サマリを返す。"""
    if df.empty:
        return pd.DataFrame(columns=["order_month", "sales_amount", "gross_profit", "net_gross_profit"])
    summary = (
        df.groupby("order_month")[
            ["sales_amount", "gross_profit", "net_gross_profit"]
        ]
        .sum()
        .reset_index()
        .sort_values("order_month")
    )
    summary["prev_year_sales"] = summary["sales_amount"].shift(12)
    summary["prev_month_sales"] = summary["sales_amount"].shift(1)
    summary["sales_yoy"] = np.where(
        (summary["prev_year_sales"].notna()) & (summary["prev_year_sales"] != 0),
        (summary["sales_amount"] - summary["prev_year_sales"]) / summary["prev_year_sales"],
        np.nan,
    )
    summary["sales_mom"] = np.where(
        (summary["prev_month_sales"].notna()) & (summary["prev_month_sales"] != 0),
        (summary["sales_amount"] - summary["prev_month_sales"]) / summary["prev_month_sales"],
        np.nan,
    )
    return summary


def compute_channel_share(df: pd.DataFrame) -> pd.DataFrame:
    """チャネル別売上構成比。"""
    if df.empty:
        return pd.DataFrame(columns=["channel", "sales_amount"])
    return (
        df.groupby("channel")["sales_amount"].sum().reset_index().sort_values("sales_amount", ascending=False)
    )


def compute_category_share(df: pd.DataFrame) -> pd.DataFrame:
    """カテゴリ別売上構成比。"""
    if df.empty:
        return pd.DataFrame(columns=["category", "sales_amount"])
    return (
        df.groupby("category")["sales_amount"].sum().reset_index().sort_values("sales_amount", ascending=False)
    )


def calculate_kpis(
    merged_df: pd.DataFrame,
    subscription_df: Optional[pd.DataFrame],
    month: Optional[pd.Period] = None,
    overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, Optional[float]]:
    """主要KPIを計算する。"""
    overrides = overrides or {}
    if merged_df.empty:
        return {}

    if month is None:
        month = merged_df["order_month"].max()

    monthly_df = merged_df[merged_df["order_month"] == month]
    monthly_sales = float(monthly_df["sales_amount"].sum())
    monthly_gross_profit = float(monthly_df["net_gross_profit"].sum())

    sub_row = None
    if subscription_df is not None and not subscription_df.empty:
        subscription_df = subscription_df.copy()
        if "month" in subscription_df.columns:
            subscription_df["month"] = pd.PeriodIndex(subscription_df["month"], freq="M")
        sub_candidates = subscription_df[subscription_df["month"] == month]
        if not sub_candidates.empty:
            sub_row = sub_candidates.iloc[0]

    active_customers = overrides.get(
        "active_customers", float(sub_row["active_customers"]) if sub_row is not None else np.nan
    )
    new_customers = overrides.get(
        "new_customers", float(sub_row["new_customers"]) if sub_row is not None else np.nan
    )
    repeat_customers = overrides.get(
        "repeat_customers", float(sub_row["repeat_customers"]) if sub_row is not None else np.nan
    )
    cancelled = overrides.get(
        "cancelled_subscriptions", float(sub_row["cancelled_subscriptions"]) if sub_row is not None else np.nan
    )
    prev_active = overrides.get(
        "previous_active_customers",
        float(sub_row.get("previous_active_customers", np.nan)) if sub_row is not None else np.nan,
    )
    marketing_cost = overrides.get(
        "marketing_cost", float(sub_row["marketing_cost"]) if sub_row is not None else np.nan
    )
    ltv_value = overrides.get("ltv", float(sub_row["ltv"]) if sub_row is not None else np.nan)

    arpu = monthly_sales / active_customers if active_customers else np.nan
    repeat_rate = repeat_customers / active_customers if active_customers else np.nan
    churn_rate = cancelled / prev_active if prev_active else np.nan
    roas = monthly_sales / marketing_cost if marketing_cost else np.nan
    adv_ratio = marketing_cost / monthly_sales if monthly_sales else np.nan
    gross_margin_rate = monthly_gross_profit / monthly_sales if monthly_sales else np.nan
    cac = marketing_cost / new_customers if new_customers else np.nan

    inventory_turnover_days = overrides.get("inventory_turnover_days", np.nan)
    stockout_rate = overrides.get("stockout_rate", np.nan)
    training_sessions = overrides.get("training_sessions", np.nan)
    new_product_count = overrides.get("new_product_count", np.nan)

    return {
        "month": month,
        "sales": monthly_sales,
        "gross_profit": monthly_gross_profit,
        "active_customers": active_customers,
        "new_customers": new_customers,
        "repeat_customers": repeat_customers,
        "cancelled_subscriptions": cancelled,
        "previous_active_customers": prev_active,
        "marketing_cost": marketing_cost,
        "ltv": ltv_value,
        "arpu": arpu,
        "repeat_rate": repeat_rate,
        "churn_rate": churn_rate,
        "roas": roas,
        "adv_ratio": adv_ratio,
        "gross_margin_rate": gross_margin_rate,
        "cac": cac,
        "inventory_turnover_days": inventory_turnover_days,
        "stockout_rate": stockout_rate,
        "training_sessions": training_sessions,
        "new_product_count": new_product_count,
    }


def annotate_customer_segments(
    df: pd.DataFrame,
    *,
    dormancy_days: int = DEFAULT_DORMANCY_DAYS,
) -> pd.DataFrame:
    """顧客ごとの購入履歴からセグメント情報を付与する。"""

    if df is None:
        return pd.DataFrame(
            columns=[
                "customer_segment",
                "is_new_customer",
                "is_reactivated_customer",
            ]
        )

    if df.empty:
        annotated = df.copy()
        annotated["customer_segment"] = pd.Series(dtype="object")
        annotated["is_new_customer"] = pd.Series(dtype="bool")
        annotated["is_reactivated_customer"] = pd.Series(dtype="bool")
        return annotated

    required_columns = {"customer_id", "order_date"}
    if not required_columns.issubset(df.columns):
        annotated = df.copy()
        annotated["customer_segment"] = "未分類"
        annotated["is_new_customer"] = False
        annotated["is_reactivated_customer"] = False
        return annotated

    working = df.sort_values(["customer_id", "order_date"]).copy()
    working["first_order_date"] = working.groupby("customer_id")["order_date"].transform("min")
    working["previous_order_date"] = working.groupby("customer_id")["order_date"].shift(1)
    working["days_since_prev"] = (
        working["order_date"] - working["previous_order_date"]
    ).dt.days

    working["customer_segment"] = np.select(
        [
            working["order_date"] == working["first_order_date"],
            working["days_since_prev"] > dormancy_days,
            working["days_since_prev"].notna(),
        ],
        ["新規", "休眠", "既存"],
        default="既存",
    )
    working.loc[working["customer_segment"].isna(), "customer_segment"] = "既存"
    working["is_new_customer"] = working["customer_segment"] == "新規"
    working["is_reactivated_customer"] = working["customer_segment"] == "休眠"
    return working


def _to_valid_float(value: Optional[float]) -> float:
    """Helper to safely convert optional values to float."""

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(converted):
        return float("nan")
    return converted


def compute_kpi_breakdown(
    df: pd.DataFrame,
    dimension: str,
    *,
    kpi_totals: Optional[Dict[str, Optional[float]]] = None,
    dormancy_days: int = DEFAULT_DORMANCY_DAYS,
) -> pd.DataFrame:
    """指定したディメンションで主要KPIを集計する。"""

    if df is None:
        return pd.DataFrame()

    working = df.copy()
    if working.empty:
        return pd.DataFrame(
            columns=[
                dimension,
                "sales_amount",
                "gross_profit",
                "orders",
                "active_customers",
                "new_customers",
                "repeat_customers",
                "reactivated_customers",
                "marketing_cost",
                "roas",
                "cac",
                "repeat_rate",
                "churn_rate",
                "ltv",
                "arpu",
                "gross_margin_rate",
                "avg_order_value",
                "sales_share",
                "profit_contribution",
                "profit_per_customer",
            ]
        )

    if "customer_segment" not in working.columns:
        working = annotate_customer_segments(working, dormancy_days=dormancy_days)

    if dimension not in working.columns:
        return pd.DataFrame()

    working[dimension] = working[dimension].fillna("未分類").astype(str)

    profit_column = None
    for candidate in ["net_gross_profit", "gross_profit"]:
        if candidate in working.columns:
            profit_column = candidate
            break

    records: List[Dict[str, Any]] = []
    for value, group_df in working.groupby(dimension):
        sales = float(group_df["sales_amount"].sum()) if "sales_amount" in group_df else 0.0
        gross_profit = (
            float(group_df[profit_column].sum())
            if profit_column and profit_column in group_df
            else float("nan")
        )
        orders = int(group_df.shape[0])
        active_customers = int(group_df["customer_id"].nunique()) if "customer_id" in group_df else 0
        new_customers = (
            int(group_df.loc[group_df["customer_segment"] == "新規", "customer_id"].nunique())
            if "customer_segment" in group_df
            else 0
        )
        reactivated_customers = (
            int(group_df.loc[group_df["customer_segment"] == "休眠", "customer_id"].nunique())
            if "customer_segment" in group_df
            else 0
        )
        repeat_customers = max(active_customers - new_customers, 0)

        records.append(
            {
                dimension: value,
                "sales_amount": sales,
                "gross_profit": gross_profit,
                "orders": orders,
                "active_customers": active_customers,
                "new_customers": new_customers,
                "repeat_customers": repeat_customers,
                "reactivated_customers": reactivated_customers,
                "share_basis_new": new_customers,
                "share_basis_sales": sales,
            }
        )

    breakdown = pd.DataFrame(records)
    if breakdown.empty:
        return breakdown

    totals = kpi_totals or {}
    marketing_total = _to_valid_float(totals.get("marketing_cost"))
    cancelled_total = _to_valid_float(totals.get("cancelled_subscriptions"))
    prev_active_total = _to_valid_float(totals.get("previous_active_customers"))
    overall_active_total = _to_valid_float(totals.get("active_customers"))

    total_sales = breakdown["sales_amount"].sum()
    breakdown["sales_share"] = (
        breakdown["sales_amount"] / total_sales if total_sales else float("nan")
    )

    if np.isfinite(marketing_total) and marketing_total > 0:
        total_new = breakdown["share_basis_new"].sum()
        if total_new > 0:
            breakdown["marketing_cost"] = marketing_total * (
                breakdown["share_basis_new"] / total_new
            )
        else:
            total_sales_basis = breakdown["share_basis_sales"].sum()
            if total_sales_basis > 0:
                breakdown["marketing_cost"] = marketing_total * (
                    breakdown["share_basis_sales"] / total_sales_basis
                )
            else:
                breakdown["marketing_cost"] = marketing_total / len(breakdown)
    else:
        breakdown["marketing_cost"] = float("nan")

    active_share_base = (
        overall_active_total
        if np.isfinite(overall_active_total) and overall_active_total > 0
        else breakdown["active_customers"].sum()
    )

    if (
        np.isfinite(cancelled_total)
        and cancelled_total >= 0
        and np.isfinite(prev_active_total)
        and prev_active_total > 0
        and active_share_base > 0
    ):
        active_share = breakdown["active_customers"] / active_share_base
        breakdown["estimated_cancelled"] = cancelled_total * active_share
        prev_active_alloc = prev_active_total * active_share
        breakdown["churn_rate"] = np.where(
            prev_active_alloc > 0,
            breakdown["estimated_cancelled"] / prev_active_alloc,
            float("nan"),
        )
    else:
        breakdown["churn_rate"] = float("nan")

    breakdown["gross_margin_rate"] = np.where(
        breakdown["sales_amount"] != 0,
        breakdown["gross_profit"] / breakdown["sales_amount"],
        float("nan"),
    )
    breakdown["avg_order_value"] = np.where(
        breakdown["orders"] != 0,
        breakdown["sales_amount"] / breakdown["orders"],
        float("nan"),
    )
    breakdown["arpu"] = np.where(
        breakdown["active_customers"] != 0,
        breakdown["sales_amount"] / breakdown["active_customers"],
        float("nan"),
    )
    breakdown["repeat_rate"] = np.where(
        breakdown["active_customers"] != 0,
        breakdown["repeat_customers"] / breakdown["active_customers"],
        float("nan"),
    )
    breakdown["ltv"] = np.where(
        breakdown["new_customers"] > 0,
        breakdown["gross_profit"] / breakdown["new_customers"],
        np.where(
            breakdown["active_customers"] > 0,
            breakdown["gross_profit"] / breakdown["active_customers"],
            float("nan"),
        ),
    )
    breakdown["roas"] = np.where(
        breakdown["marketing_cost"] > 0,
        breakdown["sales_amount"] / breakdown["marketing_cost"],
        float("nan"),
    )
    breakdown["cac"] = np.where(
        breakdown["new_customers"] > 0,
        breakdown["marketing_cost"] / breakdown["new_customers"],
        float("nan"),
    )
    breakdown["profit_contribution"] = breakdown["gross_profit"] - breakdown["marketing_cost"]
    breakdown["profit_per_customer"] = np.where(
        breakdown["active_customers"] > 0,
        breakdown["profit_contribution"] / breakdown["active_customers"],
        float("nan"),
    )

    for col in ["share_basis_new", "share_basis_sales", "estimated_cancelled"]:
        if col in breakdown.columns:
            breakdown.drop(columns=col, inplace=True)

    breakdown.sort_values("sales_amount", ascending=False, inplace=True)
    breakdown.reset_index(drop=True, inplace=True)
    return breakdown


def generate_sample_sales_data(seed: int = 42) -> pd.DataFrame:
    """分析用のサンプル売上データを生成する。"""
    rng = np.random.default_rng(seed)
    months = pd.period_range("2023-01", periods=24, freq="M")
    channels = ["自社サイト", "楽天市場", "Amazon", "Yahoo!ショッピング"]
    campaigns = ["LINE広告", "Instagram広告", "リスティング", "定期フォロー", "紹介キャンペーン"]
    stores = ["那覇本店", "浦添物流センター", "EC本部"]
    store_probabilities = [0.25, 0.25, 0.5]

    sample_products = [
        {"code": "FKD01", "name": "低分子フコイダンドリンク", "category": "フコイダン", "price": 11800, "cost_rate": 0.24},
        {"code": "FKD02", "name": "まいにちフコイダン粒", "category": "フコイダン", "price": 9800, "cost_rate": 0.26},
        {"code": "APO01", "name": "アポネクスト", "category": "サプリ", "price": 12800, "cost_rate": 0.28},
        {"code": "SMG01", "name": "玄米麹スムージー", "category": "スムージー", "price": 5400, "cost_rate": 0.45},
        {"code": "LAC01", "name": "まいにち乳酸菌", "category": "サプリ", "price": 4800, "cost_rate": 0.35},
        {"code": "BTA01", "name": "美容酢ドリンク", "category": "美容", "price": 4200, "cost_rate": 0.62},
    ]

    records: List[Dict[str, object]] = []
    for month in months:
        seasonality = 1.0 + 0.15 * math.sin((month.month - 1) / 12 * 2 * math.pi)
        for channel in channels:
            channel_bias = {
                "自社サイト": 1.0,
                "楽天市場": 0.6,
                "Amazon": 0.7,
                "Yahoo!ショッピング": 0.4,
            }[channel]
            for product in sample_products:
                demand = rng.normal(loc=120, scale=35)
                demand = max(demand, 20)
                demand *= seasonality * channel_bias
                demand *= 1 + rng.normal(0, 0.08)
                quantity = int(demand / 2)
                sales_amount = quantity * product["price"]
                customer_count = max(1, int(quantity * 0.6))
                for i in range(max(1, customer_count // 3)):
                    campaign = campaigns[rng.integers(0, len(campaigns))]
                    store = rng.choice(stores, p=store_probabilities)
                    records.append(
                        {
                            "order_date": month.to_timestamp("M") - pd.Timedelta(days=rng.integers(0, 27)),
                            "channel": channel,
                            "store": store,
                            "product_code": product["code"],
                            "product_name": product["name"],
                            "category": product["category"],
                            "quantity": quantity / max(1, customer_count // 3),
                            "sales_amount": sales_amount / max(1, customer_count // 3),
                            "customer_id": f"{channel[:2]}-{month.strftime('%Y%m')}-{rng.integers(1000, 9999)}",
                            "campaign": campaign,
                        }
                    )

    sales_df = pd.DataFrame(records)
    sales_df["order_date"] = pd.to_datetime(sales_df["order_date"])
    sales_df = normalize_sales_df(sales_df)
    # サンプル用のカテゴリ情報が失われないよう補正
    category_map = {p["name"]: p["category"] for p in sample_products}
    mapped_categories = sales_df["product_name"].map(category_map)
    update_mask = sales_df["category"].eq("未分類") & mapped_categories.notna()
    sales_df.loc[update_mask, "category"] = mapped_categories[update_mask]
    return sales_df


def generate_sample_cost_data() -> pd.DataFrame:
    """サンプルの原価率データを作成する。"""
    data = [
        ("FKD01", "低分子フコイダンドリンク", "フコイダン", 11800, 11800 * 0.24, 0.24),
        ("FKD02", "まいにちフコイダン粒", "フコイダン", 9800, 9800 * 0.26, 0.26),
        ("APO01", "アポネクスト", "サプリ", 12800, 12800 * 0.28, 0.28),
        ("SMG01", "玄米麹スムージー", "スムージー", 5400, 5400 * 0.45, 0.45),
        ("LAC01", "まいにち乳酸菌", "サプリ", 4800, 4800 * 0.35, 0.35),
        ("BTA01", "美容酢ドリンク", "美容", 4200, 4200 * 0.62, 0.62),
    ]
    return pd.DataFrame(data, columns=["product_code", "product_name", "category", "price", "cost", "cost_rate"])


def generate_sample_subscription_data() -> pd.DataFrame:
    """サンプルのサブスク/KPIデータを生成する。"""
    months = pd.period_range("2023-01", periods=24, freq="M")
    rng = np.random.default_rng(7)
    rows: List[Dict[str, object]] = []
    active = 2400
    for month in months:
        new_customers = rng.integers(120, 260)
        cancelled = rng.integers(70, 120)
        repeat_customers = int(active * 0.68 + rng.normal(0, 40))
        marketing_cost = 1_500_000 + rng.normal(0, 120_000)
        ltv = 68_000 + rng.normal(0, 3_000)
        total_sales = (active + new_customers - cancelled) * 12_000
        rows.append(
            {
                "month": month,
                "active_customers": active,
                "new_customers": new_customers,
                "repeat_customers": repeat_customers,
                "cancelled_subscriptions": cancelled,
                "previous_active_customers": active,
                "marketing_cost": marketing_cost,
                "ltv": ltv,
                "total_sales": total_sales,
            }
        )
        active = active + new_customers - cancelled
    return pd.DataFrame(rows)


def create_current_pl(merged_df: pd.DataFrame, subscription_df: Optional[pd.DataFrame], fixed_cost: float) -> Dict[str, float]:
    """現在のPLモデルを作成する。"""
    if merged_df.empty:
        return {
            "sales": 0.0,
            "cogs": 0.0,
            "gross_profit": 0.0,
            "sga": fixed_cost,
            "operating_profit": -fixed_cost,
        }
    monthly_summary = monthly_sales_summary(merged_df)
    latest = monthly_summary.iloc[-1]
    marketing_cost = 0.0
    if subscription_df is not None and not subscription_df.empty:
        subscription_df = subscription_df.copy()
        subscription_df["month"] = pd.PeriodIndex(subscription_df["month"], freq="M")
        match = subscription_df[subscription_df["month"] == latest["order_month"]]
        if not match.empty:
            marketing_cost = float(match.iloc[0]["marketing_cost"])
    sga = fixed_cost + marketing_cost
    cogs = latest["sales_amount"] - latest["net_gross_profit"]
    base_pl = {
        "sales": float(latest["sales_amount"]),
        "cogs": float(cogs),
        "gross_profit": float(latest["net_gross_profit"]),
        "sga": float(sga),
    }
    base_pl["operating_profit"] = base_pl["gross_profit"] - base_pl["sga"]
    return base_pl


def simulate_pl(
    base_pl: Dict[str, float],
    sales_growth_rate: float,
    cost_rate_adjustment: float,
    sga_change_rate: float,
    additional_ad_cost: float,
) -> pd.DataFrame:
    """PLシミュレーションを行い結果を返す。"""
    current_sales = base_pl.get("sales", 0.0)
    current_cogs = base_pl.get("cogs", 0.0)
    current_sga = base_pl.get("sga", 0.0)
    current_gross = base_pl.get("gross_profit", current_sales - current_cogs)

    new_sales = current_sales * (1 + sales_growth_rate)
    base_cost_ratio = current_cogs / current_sales if current_sales else 0
    new_cost_ratio = max(0, base_cost_ratio + cost_rate_adjustment)
    new_cogs = new_sales * new_cost_ratio
    new_gross = new_sales - new_cogs
    new_sga = current_sga * (1 + sga_change_rate) + additional_ad_cost
    new_operating_profit = new_gross - new_sga

    result = pd.DataFrame(
        {
            "項目": ["売上高", "売上原価", "粗利", "販管費", "営業利益"],
            "現状": [current_sales, current_cogs, current_gross, current_sga, current_gross - current_sga],
            "シナリオ": [new_sales, new_cogs, new_gross, new_sga, new_operating_profit],
        }
    )
    result["増減"] = result["シナリオ"] - result["現状"]
    return result


def create_default_cashflow_plan(merged_df: pd.DataFrame, horizon_months: int = 6) -> pd.DataFrame:
    """簡易キャッシュフロー予測の初期値を生成する。"""
    if merged_df.empty:
        months = pd.period_range(pd.Timestamp.today(), periods=horizon_months, freq="M")
        return pd.DataFrame(
            {
                "month": months,
                "operating_cf": 0.0,
                "investment_cf": 0.0,
                "financing_cf": 0.0,
                "loan_repayment": DEFAULT_LOAN_REPAYMENT,
            }
        )

    summary = monthly_sales_summary(merged_df)
    recent_gross = summary.tail(6)["net_gross_profit"].mean()
    plan_months = pd.period_range(summary["order_month"].iloc[-1] + 1, periods=horizon_months, freq="M")
    operating_cf = recent_gross * 0.75

    plan_df = pd.DataFrame(
        {
            "month": plan_months,
            "operating_cf": operating_cf,
            "investment_cf": -250_000,
            "financing_cf": 0.0,
            "loan_repayment": DEFAULT_LOAN_REPAYMENT,
        }
    )
    return plan_df


def forecast_cashflow(plan_df: pd.DataFrame, starting_cash: float) -> pd.DataFrame:
    """キャッシュ残高推移を計算する。"""
    if plan_df.empty:
        return pd.DataFrame(columns=["month", "net_cf", "cash_balance"])
    cash = starting_cash
    records: List[Dict[str, object]] = []
    for _, row in plan_df.iterrows():
        operating_cf = float(row.get("operating_cf", 0.0))
        investment_cf = float(row.get("investment_cf", 0.0))
        financing_cf = float(row.get("financing_cf", 0.0))
        loan_repayment = float(row.get("loan_repayment", 0.0))
        net_cf = operating_cf + financing_cf - investment_cf - loan_repayment
        cash += net_cf
        records.append(
            {
                "month": row.get("month"),
                "net_cf": net_cf,
                "cash_balance": cash,
            }
        )
    forecast_df = pd.DataFrame(records)
    return forecast_df


def build_alerts(
    monthly_summary: pd.DataFrame,
    kpi_summary: Dict[str, Optional[float]],
    cashflow_forecast: pd.DataFrame,
    thresholds: Optional[Dict[str, float]] = None,
) -> List[str]:
    """アラート文言のリストを作成する。"""
    thresholds = thresholds or {
        "revenue_drop_pct": 0.3,
        "churn_rate": 0.05,
        "gross_margin_rate": 0.6,
        "cash_balance": 0,
    }
    alerts: List[str] = []

    if monthly_summary is not None and len(monthly_summary) >= 2:
        latest = monthly_summary.iloc[-1]
        prev = monthly_summary.iloc[-2]
        if prev["sales_amount"] and (latest["sales_amount"] < prev["sales_amount"] * (1 - thresholds["revenue_drop_pct"])):
            drop_pct = (latest["sales_amount"] - prev["sales_amount"]) / prev["sales_amount"]
            alerts.append(f"売上が前月比で{drop_pct:.1%}減少しています。原因分析を行ってください。")

    churn_rate = kpi_summary.get("churn_rate") if kpi_summary else None
    if churn_rate and churn_rate > thresholds["churn_rate"]:
        alerts.append(f"解約率が{churn_rate:.1%}と高水準です。定期顧客のフォローを見直してください。")

    gross_margin_rate = kpi_summary.get("gross_margin_rate") if kpi_summary else None
    if gross_margin_rate and gross_margin_rate < thresholds["gross_margin_rate"]:
        alerts.append(f"粗利率が{gross_margin_rate:.1%}と目標を下回っています。商品ミックスを確認しましょう。")

    if cashflow_forecast is not None and not cashflow_forecast.empty:
        min_balance = cashflow_forecast["cash_balance"].min()
        if min_balance < thresholds["cash_balance"]:
            alerts.append("将来の資金残高がマイナスに落ち込む見込みです。資金繰り対策を検討してください。")

    return alerts
