"""Требования Д1-Д8 к строке заказа, записанные ожиданиями Great Expectations.

Д1-Д2 относятся к файлу целиком: при нарушении порция не обрабатывается.
Д3-Д8 проверяют отдельные заказы: нарушившие строки исключаются, их номера
берутся из результата проверки (unexpected_index_list).
"""
import logging
import os

import great_expectations as gx
import pandas as pd

from delivery_risk.data.split_batches import ORDER_FIELDS

os.environ.setdefault("GX_ANALYTICS_ENABLED", "False")
logging.getLogger("great_expectations").setLevel(logging.ERROR)

STATES = ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS",
          "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC",
          "SE", "SP", "TO"]
E = gx.expectations


def build_suite(q):
    """Список (код требования, ожидание, уровень): table - стоп, row - исключение."""
    return [
        ("Д1", E.ExpectTableColumnsToMatchSet(column_set=ORDER_FIELDS, exact_match=True), "table"),
        ("Д2", E.ExpectColumnValuesToNotBeNull(column="order_id"), "table"),
        ("Д2", E.ExpectColumnValuesToBeUnique(column="order_id"), "table"),
        ("Д3", E.ExpectColumnValuesToNotBeNull(column="order_purchase_timestamp"), "row"),
        ("Д3", E.ExpectColumnValuesToNotBeNull(column="order_estimated_delivery_date"), "row"),
        ("Д4", E.ExpectColumnValuesToBeBetween(column="promise_days",
                                               min_value=q["min_promise_days"]), "row"),
        ("Д5", E.ExpectColumnValuesToBeInSet(column="customer_state", value_set=STATES), "row"),
        ("Д5", E.ExpectColumnValuesToBeInSet(column="seller_state", value_set=STATES), "row"),
        ("Д6", E.ExpectColumnValuesToBeBetween(column="n_items", min_value=1,
                                               max_value=q["max_items"]), "row"),
        ("Д7", E.ExpectColumnValuesToBeBetween(column="approval_hours", min_value=0), "row"),
        ("Д8", E.ExpectColumnValuesToBeBetween(column="items_price", min_value=0.01,
                                               max_value=q["max_items_price"]), "row"),
        ("Д8", E.ExpectColumnValuesToBeBetween(column="items_freight", min_value=0,
                                               max_value=q["max_freight"]), "row"),
        ("Д8", E.ExpectColumnValuesToBeInSet(column="payment_type",
                                             value_set=q["payment_types"]), "row"),
    ]


_CTX = None


def _batch(df):
    global _CTX
    if _CTX is None:
        _CTX = gx.get_context(mode="ephemeral")
        try:
            _CTX.variables.progress_bars = {"globally": False}
        except Exception:
            pass
        asset = _CTX.data_sources.add_pandas("orders").add_dataframe_asset("frame")
        asset.add_batch_definition_whole_dataframe("whole")
    bd = _CTX.data_sources.get("orders").get_asset("frame").get_batch_definition("whole")
    return bd.get_batch(batch_parameters={"dataframe": df})


def with_derived(df):
    """Поля, которые проверяются, но в файле не хранятся."""
    out = df.copy()
    out["promise_days"] = (out["order_estimated_delivery_date"].dt.normalize()
                           - out["order_purchase_timestamp"].dt.normalize()).dt.days
    approved = out["order_approved_at"].fillna(out["order_purchase_timestamp"])
    out["approval_hours"] = ((approved - out["order_purchase_timestamp"])
                             .dt.total_seconds() / 3600)
    return out


def run_suite(df, q):
    """Возвращает {код: множество order_id нарушителей} и признак остановки."""
    bad, stop = {}, []
    fmt = {"result_format": "COMPLETE", "unexpected_index_column_names": ["order_id"]}
    suite = build_suite(q)
    # Д1 проверяется на файле как он есть, до вычисляемых полей
    if not _batch(df).validate(suite[0][1]).success:
        return bad, ["Д1"]
    batch = _batch(with_derived(df))
    for code, exp, level in suite[1:]:
        res = batch.validate(exp, result_format=fmt)
        if res.success:
            bad.setdefault(code, set())
            continue
        if level == "table":
            stop.append(code)
            continue
        ids = {r["order_id"] for r in res.result.get("unexpected_index_list", [])}
        bad.setdefault(code, set()).update(ids)
    return bad, stop
