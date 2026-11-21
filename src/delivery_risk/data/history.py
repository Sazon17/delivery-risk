"""История продавца на момент оплаты заказа.

Метка заказа становится известна не позже обещанной даты доставки. Поэтому
в историю продавца для заказа, оплаченного в момент t, попадают только его
прошлые заказы, у которых обещанная дата наступила раньше дня t
(требование R-03). Так история не подмешивает будущие исходы ни при
обучении, ни при обработке порции.
"""
import pandas as pd

from delivery_risk.data.io import all_parts, labels_path, read_labels, read_orders


def label_consistency(orders, labels):
    """Требование Д12: у доставленного заказа есть дата вручения, вручение не
    раньше передачи перевозчику, передача не раньше покупки. Возвращает маску
    заказов, метку которых использовать нельзя."""
    lab = labels.set_index("order_id").reindex(orders["order_id"])
    buy = orders.set_index("order_id")["order_purchase_timestamp"]
    carrier = lab["order_delivered_carrier_date"]
    customer = lab["order_delivered_customer_date"]
    no_date = (lab["order_status"] == "delivered") & customer.isna()
    order_bad = customer < carrier
    early = carrier < buy
    return (no_date | order_bad | early).to_numpy()


def load_events(parts=None):
    """Созревшие исходы: продавец, момент созревания, опоздание."""
    rows = []
    for p in parts or all_parts():
        o = read_orders(p)
        lab = read_labels(labels_path(p))
        bad = label_consistency(o, lab)
        m = o[["order_id", "seller_id", "order_estimated_delivery_date"]].merge(
            lab[["order_id", "is_late"]], on="order_id")
        m = m[~bad & m["is_late"].notna() & m["seller_id"].notna()]
        rows.append(m)
    ev = pd.concat(rows, ignore_index=True)
    # исход известен к концу обещанного дня
    ev["matured_at"] = ev["order_estimated_delivery_date"].dt.normalize() + pd.Timedelta(days=1)
    ev["is_late"] = ev["is_late"].astype(int)
    ev = ev.sort_values(["matured_at", "order_id"])
    ev["n"] = ev.groupby("seller_id").cumcount() + 1
    ev["late"] = ev.groupby("seller_id")["is_late"].cumsum()
    return ev[["seller_id", "matured_at", "n", "late"]]


def seller_history(orders, events):
    """Число созревших заказов продавца и число опозданий среди них на момент
    оплаты каждого заказа (поиск последнего события не позже покупки)."""
    left = orders[["order_id", "seller_id", "order_purchase_timestamp"]].copy()
    left["_pos"] = range(len(left))
    known = left["seller_id"].notna()
    res = pd.DataFrame({"seller_n_matured": 0.0, "seller_n_late": 0.0},
                       index=range(len(left)))
    if known.any():
        lk = left[known].sort_values("order_purchase_timestamp")
        m = pd.merge_asof(lk, events.rename(columns={"matured_at": "order_purchase_timestamp"}),
                          on="order_purchase_timestamp", by="seller_id",
                          direction="backward", allow_exact_matches=True)
        res.loc[m["_pos"].to_numpy(), "seller_n_matured"] = m["n"].fillna(0).to_numpy()
        res.loc[m["_pos"].to_numpy(), "seller_n_late"] = m["late"].fillna(0).to_numpy()
    res.index = orders.index
    return res
