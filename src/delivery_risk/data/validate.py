"""Этап validate: требования Д1-Д13 к исходной части и каждой порции.

Д1-Д8 исполняет Great Expectations (expectations.py), Д9-Д13 считаются здесь.
Результат - reports/validation.json; замеры времени пишутся отдельно в
reports/timing.json, за которым DVC не следит (иначе dvc.lock менялся бы
при каждом запуске).
"""
import sys
from pathlib import Path

if __package__ in (None, ""):  # запуск файлом: python src/delivery_risk/...
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from delivery_risk.data.expectations import run_suite, with_derived
from delivery_risk.data.history import label_consistency, load_events, seller_history
from delivery_risk.data.io import all_parts, labels_path, read_labels, read_orders

GEO = "data/raw/olist/olist_geolocation_dataset.csv"


def geo_prefixes():
    g = pd.read_csv(GEO, usecols=["geolocation_zip_code_prefix"],
                    dtype={"geolocation_zip_code_prefix": str})
    return set(g["geolocation_zip_code_prefix"])


def check_orders(df, q):
    """Д1-Д8. Возвращает (очищенная таблица, отчёт). При нарушении Д1/Д2
    очищенная таблица пустая и в отчёте stop."""
    bad, stop = run_suite(df, q)
    rep = {"rows": int(len(df)), "stop": stop,
           "violations": {k: len(v) for k, v in sorted(bad.items())}}
    if stop:
        rep["excluded"] = int(len(df))
        return df.iloc[0:0].copy(), rep
    excluded = set().union(*bad.values()) if bad else set()
    clean = df[~df["order_id"].isin(excluded)].copy()
    rep["excluded"] = int(len(excluded))
    # Д7: пустая дата подтверждения заменяется датой покупки, замены считаются
    empty = clean["order_approved_at"].isna()
    rep["approved_filled"] = int(empty.sum())
    clean.loc[empty, "order_approved_at"] = clean.loc[empty, "order_purchase_timestamp"]
    d = with_derived(clean)
    rep["long_promise"] = int((d["promise_days"] > q["long_promise_days"]).sum())
    rep["long_promise_warn"] = bool(rep["long_promise"] > q["long_promise_share"] * len(d))
    return clean, rep


def main():
    q = yaml.safe_load(open("configs/params.yaml"))["quality"]
    prefixes = geo_prefixes()
    events = load_events()
    out, timing = {}, {}
    train_new_share = None
    prev_rates = []
    for p in all_parts():
        t0 = time.perf_counter()
        orders = read_orders(p)
        clean, rep = check_orders(orders, q)
        # Д9-Д10: справочник координат и категория
        no_geo = (~clean["customer_zip_code_prefix"].isin(prefixes)
                  | ~clean["seller_zip_code_prefix"].isin(prefixes))
        rep["no_geo_share"] = round(float(no_geo.mean()), 4)
        rep["no_category_share"] = round(float(clean["product_category_name"].isna().mean()), 4)
        # Д11: доля заказов продавцов без истории
        hist = seller_history(clean, events)
        new_share = float((hist["seller_n_matured"] < q["min_seller_history"]).mean())
        rep["new_seller_share"] = round(new_share, 4)
        if train_new_share is None:
            train_new_share = new_share
        rep["new_seller_warn"] = bool(new_share > q["new_seller_ratio"] * train_new_share)
        # Д12-Д13: метки созревшей порции
        lab = read_labels(labels_path(p))
        bad_lab = label_consistency(orders, lab)
        rep["label_violations"] = int(bad_lab.sum())
        m = orders[["order_id", "order_purchase_timestamp"]].merge(lab, on="order_id")
        m = m[~bad_lab & m["is_late"].notna()]
        rep["labelled"] = int(len(m))
        rep["late_rate"] = round(float(m["is_late"].mean()), 4)
        if p.stem == "initial":
            by_month = m.groupby(m["order_purchase_timestamp"].dt.to_period("M"))["is_late"].mean()
            prev_rates = list(by_month.iloc[-3:])
        else:
            jump = abs(rep["late_rate"] - float(np.mean(prev_rates))) * 100
            rep["late_jump_pp"] = round(jump, 1)
            rep["late_jump_warn"] = bool(jump > q["late_rate_jump_pp"])
            prev_rates = prev_rates[1:] + [rep["late_rate"]]
        out[p.name] = rep
        timing[p.name] = round(time.perf_counter() - t0, 2)
        print(p.name, rep)
    Path("reports").mkdir(exist_ok=True)
    json.dump(out, open("reports/validation.json", "w"), ensure_ascii=False, indent=1)
    json.dump(timing, open("reports/timing.json", "w"), indent=1)


if __name__ == "__main__":
    main()
