"""Чтение файлов заказов и меток с правильными типами."""
from pathlib import Path

import pandas as pd

ORDER_DATES = ["order_purchase_timestamp", "order_approved_at",
               "order_estimated_delivery_date", "shipping_limit_date"]
LABEL_DATES = ["order_delivered_carrier_date", "order_delivered_customer_date"]
STR_COLS = {"customer_zip_code_prefix": str, "seller_zip_code_prefix": str,
            "order_id": str, "seller_id": str, "customer_id": str,
            "customer_unique_id": str}


def read_orders(path):
    df = pd.read_csv(path, dtype=STR_COLS)
    for c in ORDER_DATES:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def read_labels(path):
    lab = pd.read_csv(path, dtype={"order_id": str})
    for c in LABEL_DATES:
        lab[c] = pd.to_datetime(lab[c])
    lab["is_late"] = lab["is_late"].astype("Int64")
    return lab


def labels_path(orders_path):
    """data/incoming/batch_03.csv -> data/incoming/labels_03.csv,
    data/interim/initial.csv -> data/interim/labels_00.csv"""
    p = Path(orders_path)
    if p.stem == "initial":
        return p.with_name("labels_00.csv")
    return p.with_name(p.name.replace("batch_", "labels_"))


def all_parts():
    """Все файлы заказов в порядке поступления: исходная часть и порции."""
    parts = [Path("data/interim/initial.csv")]
    parts += sorted(Path("data/incoming").glob("batch_*.csv"))
    return parts
