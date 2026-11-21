"""Повторяемость подготовки данных и отсутствие заглядывания в будущее."""
import numpy as np
import pandas as pd
import pytest

from delivery_risk.features.pipeline import AddFeatures, make_pipeline
from delivery_risk.features.prepare import prepare
from delivery_risk.models.split import load_initial, monthly_folds, train_test


def test_prepare_is_repeatable():
    a = prepare("data/incoming/batch_02.csv")
    b = prepare("data/incoming/batch_02.csv")
    pd.testing.assert_frame_equal(a, b)


def test_pipeline_output_is_repeatable():
    df = load_initial().head(5000)
    y = df["is_late"].to_numpy()

    def features():
        pipe = make_pipeline(None)
        f = pipe.named_steps["features"].fit(df, y)
        return pipe.named_steps["prep"].fit_transform(f.transform(df))

    assert np.array_equal(features(), features())


def test_seller_history_uses_only_matured_labels():
    df = pd.read_parquet("data/processed/initial.parquet")
    lab = df[df["is_late"].notna()]
    for _, r in lab.sample(300, random_state=0).iterrows():
        same = lab[lab["seller_id"] == r["seller_id"]]
        ready = int((same["label_ready_at"] <= r["order_purchase_timestamp"]).sum())
        assert r["seller_n_matured"] <= ready


def test_folds_do_not_peek():
    tr, _ = train_test(load_initial(), "2018-02-01")
    for a, b in monthly_folds(tr, ["2017-11", "2018-01"]):
        start = tr["order_purchase_timestamp"].iloc[b].min().replace(day=1).normalize()
        assert (tr["label_ready_at"].iloc[a] <= start).all()
        assert (tr["order_purchase_timestamp"].iloc[a] < start).all()


def test_unknown_category_in_batch_is_ignored():
    df = load_initial().head(3000)
    from sklearn.linear_model import LogisticRegression
    m = make_pipeline(LogisticRegression(max_iter=2000)).fit(df, df["is_late"])
    new = df.head(5).copy()
    new["product_category_name"] = "категория_которой_не_было"
    p = m.predict_proba(new)[:, 1]
    assert np.isfinite(p).all()
