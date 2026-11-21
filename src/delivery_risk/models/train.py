"""Этап train: выбор итоговой модели по правилу и её обучение.

Правило (п. 1.4 отчёта ПР3). Основная метрика - среднее по пяти блокам
отношение PR-AUC к доле опозданий блока (lift). Модели, отстающие от
лучшей меньше чем на select_tolerance, считаются равными по ней; из них
берётся модель с наибольшей полнотой при точности не ниже 0,25, а при
разнице полноты меньше 0,005 - более простая.
Итоговая модель обучается на всех заказах обучающего периода, метка
которых созрела к началу тестового периода, и сохраняется вместе с
описанием, по которому её можно воспроизвести.
"""
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # запуск файлом: python src/delivery_risk/...
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from delivery_risk.features.pipeline import make_pipeline
from delivery_risk.models.split import load_initial, matured_by, train_test

# чем меньше номер, тем модель проще и быстрее
SIMPLICITY = {"linear": 0, "nb": 1, "knn": 2, "tree": 3, "rf": 4, "gb": 5}


def choose(exp, tol):
    cand = exp[exp["family"] != "base"].copy()
    best = cand["lift"].max()
    close = cand[cand["lift"] >= best - tol].copy()
    top_recall = close["recall_p25"].max()
    close = close[close["recall_p25"] >= top_recall - 0.005].copy()
    close["simple"] = close["family"].map(SIMPLICITY)
    return close.sort_values(["simple", "recall_p25"], ascending=[True, False]).iloc[0], best


def build(row, rs):
    p = json.loads(row["params"])
    drop = tuple(p.pop("drop", ()))
    fs = "base" if row["features"] == "base" else "full"
    if row["family"] == "linear":
        cw = p.get("class_weight")
        cw = None if cw in (None, "None") else cw
        model = LogisticRegression(C=p["C"], class_weight=cw, max_iter=5000, random_state=rs)
        return make_pipeline(model, feature_set=fs, drop=drop)
    if row["family"] == "gb":
        model = HistGradientBoostingClassifier(early_stopping=False, random_state=rs, **p) if p \
            else HistGradientBoostingClassifier(random_state=rs)
        return make_pipeline(model, feature_set=fs, drop=drop, scale=False)
    if row["family"] == "rf":
        model = RandomForestClassifier(random_state=rs, n_jobs=-1, **p)
        return make_pipeline(model, feature_set=fs, drop=drop, scale=False)
    raise ValueError("семейство не предусмотрено: " + row["family"])


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main():
    P = yaml.safe_load(open("configs/params.yaml"))["train"]
    exp = pd.read_csv("reports/experiments.csv")
    row, best = choose(exp, P["select_tolerance"])
    df = load_initial()
    train, test = train_test(df, P["test_start"])
    fit_rows = train[matured_by(train, P["test_start"])]
    model = build(row, P["random_state"])
    model.fit(fit_rows, fit_rows["is_late"].to_numpy())
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/model_v1.joblib")
    info = {
        "version": "v1",
        "trained_as_of": P["test_start"],
        "data": {"file": "data/processed/initial.parquet",
                 "md5": md5("data/processed/initial.parquet"),
                 "orders_purchased_before": P["test_start"],
                 "rows_used": int(len(fit_rows)), "late_rate": round(float(fit_rows["is_late"].mean()), 4),
                 "rows_not_matured": int(len(train) - len(fit_rows))},
        "test": {"rows": int(len(test)), "from": P["test_start"], "used_in": "ПР4"},
        "experiment": row["name"],
        "family": row["family"],
        "params": json.loads(row["params"]),
        "cv": {"scheme": "помесячные блоки " + ", ".join(P["cv_months"]),
               "lift": row["lift"], "lift_std": row["lift_std"], "pr_auc": row["pr_auc"],
               "roc_auc": row["roc_auc"], "recall_at_precision_0.25": row["recall_p25"],
               "best_lift_all_models": float(best)},
        "random_state": P["random_state"],
        "python_packages": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                            "numpy": np.__version__},
        "features_step": "src/delivery_risk/features/pipeline.py",
        "prepare_step": "src/delivery_risk/features/prepare.py",
    }
    json.dump(info, open("models/model_v1.json", "w"), ensure_ascii=False, indent=1)
    print("выбрана модель:", row["name"], "lift", row["lift"], "(лучший", best, ")")
    print("обучена на", len(fit_rows), "заказах")


if __name__ == "__main__":
    main()
