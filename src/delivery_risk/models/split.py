"""Обучающая и тестовая выборки, помесячная кросс-валидация, метрики.

Деление по времени: заказы до test_start - обучающая выборка, после -
отложенная тестовая (используется только в ПР4). Проверочный блок -
один календарный месяц; в обучение блока попадают только заказы, метка
которых созрела к началу этого месяца (обещанная дата уже прошла). Так
модель на каждом блоке знает ровно то, что знала бы в работе.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score

PROCESSED = "data/processed/initial.parquet"


def load_initial(path=PROCESSED):
    df = pd.read_parquet(path)
    return df[df["is_late"].notna()].reset_index(drop=True).assign(
        is_late=lambda d: d["is_late"].astype(int))


def train_test(df, test_start):
    t = pd.Timestamp(test_start)
    tr = df[df["order_purchase_timestamp"] < t].reset_index(drop=True)
    te = df[df["order_purchase_timestamp"] >= t].reset_index(drop=True)
    return tr, te


def matured_by(df, moment):
    """Заказы, метка которых известна к моменту moment."""
    return (df["label_ready_at"] <= pd.Timestamp(moment)).to_numpy()


def monthly_folds(df, months):
    """Список (обучающие индексы, проверочные индексы) по месяцам."""
    folds = []
    buy = df["order_purchase_timestamp"]
    for m in months:
        start = pd.Timestamp(m + "-01")
        end = start + pd.offsets.MonthBegin(1)
        tr = np.where((buy < start).to_numpy() & matured_by(df, start))[0]
        va = np.where(((buy >= start) & (buy < end)).to_numpy())[0]
        folds.append((tr, va))
    return folds


def recall_at_precision(y, p, min_precision):
    prec, rec, _ = precision_recall_curve(y, p)
    ok = prec >= min_precision
    return float(rec[ok].max()) if ok.any() else 0.0


def scores(y, p, min_precision=0.25):
    base = float(np.mean(y))
    ap = float(average_precision_score(y, p))
    return {"pr_auc": ap, "lift": ap / base, "roc_auc": float(roc_auc_score(y, p)),
            "recall_p25": recall_at_precision(y, p, min_precision),
            "brier": float(brier_score_loss(y, p)), "base_rate": base}
