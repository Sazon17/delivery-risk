"""Подготовка любого файла заказов (исходной части или порции) для модели.

Одна команда:
    python src/delivery_risk/features/prepare.py data/incoming/batch_01.csv
читает файл, применяет проверки Д1-Д8 и очистку, добавляет расстояние,
историю продавца и загрузку площадки и сохраняет таблицу в data/processed.

Здесь только шаги без параметров, оцениваемых по данным. Всё, что
учится на обучающей выборке (медианы срока по штату, общая доля
опозданий), находится в transformers.py внутри Pipeline и при
кросс-валидации оценивается заново на каждом обучающем блоке.
"""
import sys
from pathlib import Path

if __package__ in (None, ""):  # запуск файлом: python src/delivery_risk/...
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import yaml

from delivery_risk.data.geo import distance_km, load_reference
from delivery_risk.data.history import label_consistency, load_events, seller_history
from delivery_risk.data.io import all_parts, labels_path, read_labels, read_orders
from delivery_risk.data.validate import check_orders

KEEP = ["order_id", "seller_id", "order_purchase_timestamp", "order_estimated_delivery_date",
        "customer_state", "seller_state", "product_category_name", "payment_type",
        "n_items", "n_sellers", "items_price", "items_freight", "weight_g_total",
        "volume_cm3_total", "payment_installments", "payment_value"]


def platform_load():
    """Число заказов площадки за 7 предыдущих дней (тыс.) по дате покупки.
    Время покупки известно сразу, метки здесь не нужны."""
    days = pd.concat([read_orders(p)["order_purchase_timestamp"].dt.normalize()
                      for p in all_parts()])
    daily = days.value_counts().sort_index()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()), fill_value=0)
    return (daily.rolling(7, min_periods=1).sum().shift(1).fillna(0) / 1000).rename("load_7d")


def prepare(path, q=None, ref=None, events=None, load=None):
    """Файл заказов -> таблица, готовая к подаче в Pipeline."""
    q = q or yaml.safe_load(open("configs/params.yaml"))["quality"]
    ref = ref if ref is not None else load_reference()
    events = events if events is not None else load_events()
    load = load if load is not None else platform_load()

    clean, rep = check_orders(read_orders(path), q)
    if rep["stop"]:
        raise ValueError(f"{path}: порция не обрабатывается, нарушены {rep['stop']}")
    df = clean[KEEP].reset_index(drop=True)
    buy = clean["order_purchase_timestamp"].reset_index(drop=True)
    df["product_category_name"] = df["product_category_name"].fillna("unknown")
    df["promise_days"] = (df["order_estimated_delivery_date"].dt.normalize() - buy.dt.normalize()).dt.days
    df["approval_h"] = ((clean["order_approved_at"].reset_index(drop=True) - buy)
                        .dt.total_seconds() / 3600)
    # срок передачи перевозчику: целые сутки от покупки, не больше 30 (Таблица 1.3 ПР2)
    ship = (clean["shipping_limit_date"].reset_index(drop=True) - buy).dt.days
    df["ship_limit_days"] = ship.clip(upper=30)
    df["dist_km"], df["geo_by_state"] = distance_km(clean.reset_index(drop=True), ref)
    df["geo_by_state"] = df["geo_by_state"].astype(int)
    hist = seller_history(clean.reset_index(drop=True), events)
    df["seller_n_matured"] = hist["seller_n_matured"].to_numpy()
    df["seller_n_late"] = hist["seller_n_late"].to_numpy()
    df["load_7d"] = buy.dt.normalize().map(load).fillna(0).to_numpy()
    return df


def attach_labels(df, path):
    """Метка и момент её созревания. Метки с нарушением Д12 не используются."""
    o = read_orders(path)
    lab = read_labels(labels_path(path))
    bad = pd.Series(label_consistency(o, lab), index=o["order_id"])
    m = df[["order_id"]].merge(lab[["order_id", "is_late"]], on="order_id", how="left")
    y = m["is_late"].astype("Float64")
    y[bad.reindex(df["order_id"]).fillna(False).to_numpy()] = pd.NA
    out = df.copy()
    out["is_late"] = y.to_numpy()
    out["label_ready_at"] = df["order_estimated_delivery_date"].dt.normalize() + pd.Timedelta(days=1)
    return out


def main(paths):
    q = yaml.safe_load(open("configs/params.yaml"))["quality"]
    ref, events, load = load_reference(), load_events(), platform_load()
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        p = Path(p)
        df = attach_labels(prepare(p, q, ref, events, load), p)
        df.to_parquet(out_dir / (p.stem + ".parquet"), index=False)
        print(f"{p.name}: {len(df)} заказов, с меткой {int(df['is_late'].notna().sum())}")


if __name__ == "__main__":
    main(sys.argv[1:] or [str(p) for p in all_parts()])
