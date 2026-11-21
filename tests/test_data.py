"""Отрицательные случаи проверок Д1-Д8 на маленьком фрагменте порции."""
import pandas as pd
import pytest
import yaml

from delivery_risk.data.io import read_orders
from delivery_risk.data.validate import check_orders

Q = yaml.safe_load(open("configs/params.yaml"))["quality"]


@pytest.fixture(scope="module")
def sample():
    df = read_orders("data/incoming/batch_01.csv")
    return df[df["n_items"] > 0].head(50).reset_index(drop=True)


def test_clean_sample_passes(sample):
    clean, rep = check_orders(sample, Q)
    assert rep["stop"] == [] and len(clean) == len(sample)


@pytest.mark.parametrize("col,value", [
    ("customer_state", "XX"),
    ("items_price", 0.0),
])
def test_bad_value_excludes_one_row(sample, col, value):
    bad = sample.copy()
    bad.loc[3, col] = value
    clean, rep = check_orders(bad, Q)
    assert rep["excluded"] == 1 and len(clean) == len(sample) - 1


def test_promise_before_purchase_excluded(sample):
    bad = sample.copy()
    bad.loc[5, "order_estimated_delivery_date"] = bad.loc[5, "order_purchase_timestamp"] - pd.Timedelta(days=2)
    _, rep = check_orders(bad, Q)
    assert rep["excluded"] == 1


def test_long_promise_counted_not_excluded(sample):
    bad = sample.copy()
    bad.loc[7, "order_estimated_delivery_date"] = bad.loc[7, "order_purchase_timestamp"] + pd.Timedelta(days=200)
    clean, rep = check_orders(bad, Q)
    assert rep["excluded"] == 0 and rep["long_promise"] >= 1


def test_label_field_in_batch_stops(sample):
    bad = sample.copy()
    bad["order_delivered_customer_date"] = pd.NaT
    clean, rep = check_orders(bad, Q)
    assert rep["stop"] == ["Д1"] and clean.empty


def test_duplicate_order_id_stops(sample):
    bad = pd.concat([sample, sample.head(1)], ignore_index=True)
    _, rep = check_orders(bad, Q)
    assert "Д2" in rep["stop"]
