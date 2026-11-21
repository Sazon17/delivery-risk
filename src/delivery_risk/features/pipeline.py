"""Признаки по плану ПР2 (Таблица 3.2) и Pipeline «признаки - предобработка - модель».

AddFeatures учит по обучающей выборке две вещи: медиану обещанного срока
по штату покупателя и общую долю опозданий для сглаживания истории
продавца. Поэтому шаг стоит внутри Pipeline: при кросс-валидации он
переобучается на каждом обучающем блоке и не видит проверочный.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CAT_COLS = ["customer_state", "seller_state", "product_category_name", "payment_type"]
# исходные числовые поля строки заказа; скошенные логарифмируются
BASE_NUM = ["promise_days", "approval_h", "ship_limit_days", "n_items", "n_sellers",
            "payment_installments", "log_items_price", "log_items_freight",
            "log_weight", "log_volume", "log_payment_value"]
# восемь новых признаков плана ПР2
NEW_NUM = ["dist_km", "promise_rel", "seller_late_rate", "seller_n_matured_log",
           "slow_approval", "heavy", "load_7d", "freight_ratio"]
SMOOTH_WEIGHT = 10   # вес априорной доли при сглаживании истории продавца


class AddFeatures(BaseEstimator, TransformerMixin):
    """Расчёт признаков. feature_set='base' - только исходные поля,
    'full' - исходные и новые; drop - признаки, исключённые при отборе."""

    def __init__(self, feature_set="full", drop=()):
        self.feature_set = feature_set
        self.drop = drop

    def fit(self, X, y=None):
        self.state_promise_ = X.groupby("customer_state")["promise_days"].median().to_dict()
        self.global_promise_ = float(X["promise_days"].median())
        self.prior_ = float(np.mean(y)) if y is not None else 0.08
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        for c in CAT_COLS:
            out[c] = X[c].astype(object)
        for c in ["promise_days", "approval_h", "ship_limit_days", "n_items", "n_sellers",
                  "payment_installments"]:
            out[c] = X[c].astype(float)
        for src, dst in [("items_price", "log_items_price"), ("items_freight", "log_items_freight"),
                         ("weight_g_total", "log_weight"), ("volume_cm3_total", "log_volume"),
                         ("payment_value", "log_payment_value")]:
            out[dst] = np.log1p(X[src].astype(float))
        if self.feature_set == "full":
            med = X["customer_state"].map(self.state_promise_).fillna(self.global_promise_)
            out["dist_km"] = X["dist_km"].astype(float)
            out["promise_rel"] = X["promise_days"] / med
            out["seller_late_rate"] = ((X["seller_n_late"] + SMOOTH_WEIGHT * self.prior_)
                                       / (X["seller_n_matured"] + SMOOTH_WEIGHT))
            out["seller_n_matured_log"] = np.log1p(X["seller_n_matured"])
            out["slow_approval"] = (X["approval_h"] > 24).astype(float)
            out["heavy"] = (X["weight_g_total"] > 5000).astype(float)
            out["load_7d"] = X["load_7d"].astype(float)
            out["freight_ratio"] = np.log1p(X["items_freight"] / X["items_price"])
        return out.drop(columns=[c for c in self.drop if c in out.columns])

    def columns(self):
        num = BASE_NUM + (NEW_NUM if self.feature_set == "full" else [])
        num = [c for c in num if c not in self.drop]
        cat = [c for c in CAT_COLS if c not in self.drop]
        return num, cat


def make_pipeline(model, feature_set="full", drop=(), scale=True):
    feats = AddFeatures(feature_set, tuple(drop))
    num, cat = feats.columns()
    numeric = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric.append(("scale", StandardScaler()))
    categorical = [("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                   ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    prep = ColumnTransformer([("num", Pipeline(numeric), num),
                              ("cat", Pipeline(categorical), cat)])
    return Pipeline([("features", feats), ("prep", prep), ("model", model)])
