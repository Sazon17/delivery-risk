"""Сборка строки заказа на момент оплаты и разбиение по месяцам оформления.

Из девяти таблиц Olist собирается одна строка на заказ - то, что учётная
система видит при подтверждении оплаты. Поля, которые появляются позже
(статус, передача перевозчику, вручение), уходят в отдельный файл меток.

Выход:
  data/interim/initial.csv, data/interim/labels_00.csv  - исходная часть
  data/incoming/batch_KK.csv, data/incoming/labels_KK.csv - порции 01..05
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ORDER_FIELDS = [
    "order_id", "customer_id", "customer_unique_id",
    "order_purchase_timestamp", "order_approved_at",
    "order_estimated_delivery_date", "shipping_limit_date",
    "customer_zip_code_prefix", "customer_city", "customer_state",
    "seller_id", "seller_zip_code_prefix", "seller_city", "seller_state",
    "product_category_name",
    "n_items", "n_products", "n_sellers", "items_price", "items_freight",
    "weight_g_total", "volume_cm3_total",
    "payment_type", "payment_installments", "payment_value", "n_payments",
]
LABEL_FIELDS = ["order_id", "order_status", "order_delivered_carrier_date",
                "order_delivered_customer_date", "is_late"]


def read_raw(raw_dir):
    raw = Path(raw_dir)
    r = lambda name, **kw: pd.read_csv(raw / f"olist_{name}_dataset.csv", **kw)
    zips = {"customer_zip_code_prefix": str, "seller_zip_code_prefix": str}
    return {
        "orders": r("orders", parse_dates=[
            "order_purchase_timestamp", "order_approved_at",
            "order_delivered_carrier_date", "order_delivered_customer_date",
            "order_estimated_delivery_date"]),
        "customers": r("customers", dtype=zips),
        "sellers": r("sellers", dtype=zips),
        "items": r("order_items", parse_dates=["shipping_limit_date"]),
        "products": r("products"),
        "payments": r("order_payments"),
    }


def build_orders(t):
    """Одна строка на заказ с полями, известными при оплате, и метка."""
    # в выгрузке покупателей и продавцов у индексов потерян ведущий ноль
    for name, col in (("customers", "customer_zip_code_prefix"),
                      ("sellers", "seller_zip_code_prefix")):
        t[name][col] = t[name][col].str.zfill(5)

    items = t["items"].merge(t["products"], on="product_id", how="left")
    items["volume_cm3"] = (items["product_length_cm"] * items["product_height_cm"]
                           * items["product_width_cm"])
    agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "size"),
        n_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        items_price=("price", "sum"),
        items_freight=("freight_value", "sum"),
        weight_g_total=("product_weight_g", lambda s: s.sum(min_count=1)),
        volume_cm3_total=("volume_cm3", lambda s: s.sum(min_count=1)),
        shipping_limit_date=("shipping_limit_date", "max"),
    )
    # продавец и категория - по самой дорогой позиции заказа
    main = (items.sort_values(["order_id", "price", "order_item_id"],
                              ascending=[True, False, True])
            .drop_duplicates("order_id")[["order_id", "seller_id", "product_category_name"]])

    pay = t["payments"]
    main_pay = (pay.sort_values(["order_id", "payment_value", "payment_sequential"],
                                ascending=[True, False, True])
                .drop_duplicates("order_id")[["order_id", "payment_type"]])
    pay_agg = pay.groupby("order_id").agg(
        payment_installments=("payment_installments", "max"),
        payment_value=("payment_value", "sum"),
        n_payments=("payment_sequential", "size"))

    df = (t["orders"]
          .merge(t["customers"], on="customer_id", how="left")
          .merge(agg, on="order_id", how="left")
          .merge(main, on="order_id", how="left")
          .merge(t["sellers"], on="seller_id", how="left")
          .merge(main_pay, on="order_id", how="left")
          .merge(pay_agg, on="order_id", how="left"))
    df["n_items"] = df["n_items"].fillna(0).astype(int)

    delivered = ((df["order_status"] == "delivered")
                 & df["order_delivered_customer_date"].notna())
    late = (df["order_delivered_customer_date"].dt.normalize()
            > df["order_estimated_delivery_date"].dt.normalize())
    df["is_late"] = np.where(delivered, late.astype(int), np.nan)
    df = df.sort_values(["order_purchase_timestamp", "order_id"]).reset_index(drop=True)
    return df


def write_part(df, orders_path, labels_path):
    orders_path.parent.mkdir(parents=True, exist_ok=True)
    df[ORDER_FIELDS].to_csv(orders_path, index=False)
    lab = df[LABEL_FIELDS].copy()
    lab["is_late"] = lab["is_late"].astype("Int64")
    lab.to_csv(labels_path, index=False)


def main():
    params = yaml.safe_load(open("configs/params.yaml"))["split"]
    df = build_orders(read_raw(params["raw_dir"]))
    month = df["order_purchase_timestamp"].dt.strftime("%Y-%m")

    initial = df[df["order_purchase_timestamp"] < params["initial_end"]]
    write_part(initial, Path("data/interim/initial.csv"),
               Path("data/interim/labels_00.csv"))
    for k, m in enumerate(params["batch_months"], start=1):
        part = df[month == m]
        write_part(part, Path(f"data/incoming/batch_{k:02d}.csv"),
                   Path(f"data/incoming/labels_{k:02d}.csv"))
        print(f"batch_{k:02d}: {m}, {len(part)} заказов")
    print(f"initial: {len(initial)} заказов, "
          f"вне разбиения: {len(df) - len(initial) - month.isin(params['batch_months']).sum()}")


if __name__ == "__main__":
    main()
