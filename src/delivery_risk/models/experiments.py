"""Этап experiments: сравнение моделей по помесячной кросс-валидации.

Каждый эксперимент - Pipeline (признаки, предобработка, модель), который
на каждом из пяти блоков обучается заново. По каждому эксперименту в
reports/experiments.csv пишутся параметры, метрики по блокам, среднее,
стандартное отклонение и время обучения; тот же набор уходит в MLflow
(локальная база mlflow.db). Тестовая выборка здесь не используется.
"""
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # запуск файлом: python src/delivery_risk/...
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from delivery_risk.features.pipeline import NEW_NUM, make_pipeline
from delivery_risk.models.split import load_initial, monthly_folds, scores, train_test

OUT = Path("reports/pr3")
METRICS = ["lift", "pr_auc", "roc_auc", "recall_p25", "brier"]


class StateRule(BaseEstimator, ClassifierMixin):
    """Базовое правило из ПР1: риск заказа = доля опозданий по штату
    покупателя в обучающем блоке."""

    def fit(self, X, y):
        s = pd.Series(np.asarray(y), index=X.index)
        self.rate_ = s.groupby(X["customer_state"]).mean().to_dict()
        self.prior_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = X["customer_state"].map(self.rate_).fillna(self.prior_).to_numpy(float)
        return np.column_stack([1 - p, p])


def _lift(y, p):
    return average_precision_score(y, p) / np.mean(y)


# тот же критерий, что и в правиле выбора: PR-AUC, делённый на долю опозданий блока
LIFT = make_scorer(_lift, response_method="predict_proba")


class Runner:
    def __init__(self, X, y, folds, params):
        self.X, self.y, self.folds, self.p = X, y, folds, params
        self.rows = []
        self.timing = {}
        self.mlflow = None
        try:
            import mlflow
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("delivery-risk-pr3")
            self.mlflow = mlflow
        except Exception as e:  # журнал MLflow необязателен, CSV пишется всегда
            print("MLflow недоступен:", e)

    def cv(self, make, train_scores=False):
        res, tr_res, fit_s = [], [], []
        for a, b in self.folds:
            model = make()
            t0 = time.perf_counter()
            model.fit(self.X.iloc[a], self.y[a])
            fit_s.append(time.perf_counter() - t0)
            res.append(scores(self.y[b], model.predict_proba(self.X.iloc[b])[:, 1]))
            if train_scores:
                tr_res.append(scores(self.y[a], model.predict_proba(self.X.iloc[a])[:, 1]))
        return res, tr_res, float(np.mean(fit_s)), model

    def run(self, name, family, make, params, features="full", note=""):
        res, _, fit_s, model = self.cv(make)
        row = {"name": name, "family": family, "features": features, "params": json.dumps(params),
               "note": note}
        self.timing[name] = round(fit_s, 2)
        for k in METRICS:
            v = [r[k] for r in res]
            row[k] = round(float(np.mean(v)), 4)
            row[k + "_std"] = round(float(np.std(v)), 4)
        row["lift_folds"] = " ".join(f"{r['lift']:.2f}" for r in res)
        self.rows.append(row)
        if self.mlflow:
            with self.mlflow.start_run(run_name=name):
                self.mlflow.log_params({"family": family, "features": features, **params})
                self.mlflow.log_metrics({k: row[k] for k in METRICS} | {"fit_s": fit_s})
                for i, r in enumerate(res):
                    self.mlflow.log_metrics({f"{k}_fold{i + 1}": r[k] for k in METRICS})
        print(f"{name:45s} lift {row['lift']:.3f}±{row['lift_std']:.3f} "
              f"pr {row['pr_auc']:.3f} roc {row['roc_auc']:.3f} rec {row['recall_p25']:.3f} {fit_s:.1f}s")
        return row, model


def lr(C, penalty="l2", cw=None, rs=42):
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    return LogisticRegression(C=C, l1_ratio=1.0 if penalty == "l1" else 0.0, solver=solver,
                              class_weight=cw, max_iter=5000, random_state=rs)


def main():
    P = yaml.safe_load(open("configs/params.yaml"))["train"]
    rs = P["random_state"]
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_initial()
    train, test = train_test(df, P["test_start"])
    X, y = train, train["is_late"].to_numpy()
    folds = monthly_folds(train, P["cv_months"])
    R = Runner(X, y, folds, P)
    summary = {"train_rows": int(len(train)), "test_rows": int(len(test)),
               "train_late": round(float(y.mean()), 4), "test_late": round(float(test["is_late"].mean()), 4),
               "folds": [{"month": m, "train": int(len(a)), "valid": int(len(b)),
                          "late": round(float(y[b].mean()), 4)} for m, (a, b) in zip(P["cv_months"], folds)]}

    # 1. базовые решения
    R.run("Константа (доля опозданий)", "base", lambda: DummyClassifier(strategy="prior"), {}, "-")
    R.run("Правило: доля опозданий штата", "base", StateRule, {}, "customer_state")

    # 2. логистическая регрессия: путь регуляризации L1 и L2
    path = []
    for C in P["lr_C"]:
        for pen in ("l2", "l1"):
            row, model = R.run(f"Логистическая регрессия {pen.upper()}, C={C}", "linear",
                               lambda C=C, pen=pen: make_pipeline(lr(C, pen, rs=rs)), {"C": C, "penalty": pen})
            coef = model.named_steps["model"].coef_.ravel()
            path.append({"C": C, "penalty": pen, "lift": row["lift"], "pr_auc": row["pr_auc"],
                         "roc_auc": row["roc_auc"], "nonzero": int((np.abs(coef) > 1e-8).sum()),
                         "n_coef": int(coef.size)})
    summary["lr_path"] = path
    # коэффициенты L2-модели на всей обучающей выборке блоков (последний блок)
    a, _ = folds[-1]
    m = make_pipeline(lr(P["lr_report_C"], "l2", rs=rs)).fit(X.iloc[a], y[a])
    names = m.named_steps["prep"].get_feature_names_out()
    coef = pd.Series(m.named_steps["model"].coef_.ravel(), index=names)
    top = coef.reindex(coef.abs().sort_values(ascending=False).index)
    summary["lr_top_coef"] = {k: round(float(v), 3) for k, v in top.head(16).items()}
    summary["lr_new_coef"] = {k: round(float(coef["num__" + k]), 3) for k in NEW_NUM}

    # 3. модели классификации разной природы
    R.run(f"k ближайших соседей, k={P['knn_k']}", "knn",
          lambda: make_pipeline(KNeighborsClassifier(n_neighbors=P["knn_k"])), {"k": P["knn_k"]})
    R.run("Наивный байесовский (гауссовский)", "nb", lambda: make_pipeline(GaussianNB()), {})
    depth_curve = []
    for d in P["tree_depths"] + [None]:
        res, tr_res, fit_s, _ = R.cv(lambda d=d: make_pipeline(
            DecisionTreeClassifier(max_depth=d, random_state=rs), scale=False), train_scores=True)
        depth_curve.append({"depth": d if d else "нет", "valid_pr_auc": float(np.mean([r["pr_auc"] for r in res])),
                            "train_pr_auc": float(np.mean([r["pr_auc"] for r in tr_res])),
                            "train_lift": float(np.mean([r["lift"] for r in tr_res])),
                            "valid_lift": float(np.mean([r["lift"] for r in res]))})
    summary["tree_depth"] = depth_curve
    best_d = max((c for c in depth_curve if c["depth"] != "нет"), key=lambda c: c["valid_lift"])["depth"]
    summary["tree_best_depth"] = best_d
    R.run(f"Дерево решений, глубина {best_d}", "tree", lambda: make_pipeline(
        DecisionTreeClassifier(max_depth=best_d, random_state=rs), scale=False), {"max_depth": best_d})
    R.run("Дерево решений без ограничения глубины", "tree", lambda: make_pipeline(
        DecisionTreeClassifier(random_state=rs), scale=False), {"max_depth": None})

    # 4. ансамбли
    R.run("Случайный лес, 300 деревьев", "rf", lambda: make_pipeline(RandomForestClassifier(
        n_estimators=300, min_samples_leaf=10, n_jobs=-1, random_state=rs), scale=False),
        {"n_estimators": 300, "min_samples_leaf": 10})
    R.run("Градиентный бустинг (по умолчанию)", "gb", lambda: make_pipeline(
        HistGradientBoostingClassifier(random_state=rs), scale=False), {})

    # 5. польза новых признаков: те же модели на исходных полях
    R.run("Логистическая регрессия L2, C=1, без новых признаков", "linear",
          lambda: make_pipeline(lr(1.0, "l2", rs=rs), feature_set="base"), {"C": 1.0}, "base")
    R.run("Градиентный бустинг, без новых признаков", "gb", lambda: make_pipeline(
        HistGradientBoostingClassifier(random_state=rs), feature_set="base", scale=False), {}, "base")

    # 6. подбор гиперпараметров (GridSearchCV на тех же помесячных блоках)
    gb_grid = {f"model__{k}": v for k, v in P["gb_grid"].items()}
    gs = GridSearchCV(make_pipeline(HistGradientBoostingClassifier(
        early_stopping=False, random_state=rs), scale=False),
                      gb_grid, scoring=LIFT, cv=folds, n_jobs=1, refit=False)
    t0 = time.perf_counter()
    gs.fit(X, y)
    cvres = pd.DataFrame(gs.cv_results_)
    gb_best = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
    R.timing["GridSearchCV бустинга, всего"] = round(time.perf_counter() - t0)
    summary["gb_search"] = {"combinations": int(len(cvres)),
                            "best": gb_best, "best_lift": round(float(gs.best_score_), 4)}
    summary["gb_search_top"] = (cvres.sort_values("rank_test_score").head(5)[["params", "mean_test_score"]]
                                .assign(params=lambda d: d["params"].astype(str)).to_dict("records"))
    gb_row, _ = R.run("Градиентный бустинг (после подбора)", "gb", lambda: make_pipeline(
        HistGradientBoostingClassifier(early_stopping=False, random_state=rs, **gb_best), scale=False), gb_best)

    lr_grid = {"model__C": P["lr_grid_C"], "model__class_weight": [None, "balanced"]}
    gs2 = GridSearchCV(make_pipeline(lr(1.0, rs=rs)), lr_grid, scoring=LIFT,
                       cv=folds, n_jobs=1, refit=False).fit(X, y)
    lr_best = {k.replace("model__", ""): v for k, v in gs2.best_params_.items()}
    summary["lr_search"] = {"combinations": len(gs2.cv_results_["params"]), "best": lr_best,
                            "results": [{"C": p["model__C"], "cw": str(p["model__class_weight"]),
                                         "lift": round(float(s), 4)} for p, s in
                                        zip(gs2.cv_results_["params"], gs2.cv_results_["mean_test_score"])]}
    R.run("Логистическая регрессия L2 (после подбора)", "linear",
          lambda: make_pipeline(lr(lr_best["C"], "l2", lr_best["class_weight"], rs=rs)),
          {"C": lr_best["C"], "class_weight": str(lr_best["class_weight"])})

    # 7. отбор признаков: фильтр (взаимная информация) и обёртка (исключение по одному)
    lr_row = R.rows[-1]
    lr_cw = None if lr_best["class_weight"] in (None, "None") else lr_best["class_weight"]
    lr_drop = {}
    for f in NEW_NUM:
        res, _, _, _ = R.cv(lambda f=f: make_pipeline(lr(lr_best["C"], "l2", lr_cw, rs=rs), drop=(f,)))
        lr_drop[f] = round(float(np.mean([r["lift"] for r in res])) - lr_row["lift"], 4)
    summary["lr_drop_one"] = lr_drop
    lr_useless = [f for f, d in lr_drop.items() if d > P["drop_gain"]]
    if lr_useless:
        R.run("Логистическая регрессия L2 (после подбора), без " + ", ".join(lr_useless), "linear",
              lambda: make_pipeline(lr(lr_best["C"], "l2", lr_cw, rs=rs), drop=tuple(lr_useless)),
              {"C": lr_best["C"], "class_weight": str(lr_best["class_weight"]), "drop": lr_useless})
    feats = make_pipeline(lr(1.0)).named_steps["features"].fit(X, y)
    F = feats.transform(X)
    num = [c for c in F.columns if F[c].dtype != object]
    mi = mutual_info_classif(F[num].fillna(F[num].median()), y, random_state=rs)
    summary["mutual_info"] = {c: round(float(v), 4) for c, v in
                              sorted(zip(num, mi), key=lambda t: -t[1])}
    drop_res = {}
    for f in NEW_NUM:
        res, _, _, _ = R.cv(lambda f=f: make_pipeline(HistGradientBoostingClassifier(
            early_stopping=False, random_state=rs, **gb_best), drop=(f,), scale=False))
        drop_res[f] = round(float(np.mean([r["lift"] for r in res])) - gb_row["lift"], 4)
    summary["drop_one"] = drop_res
    useless = [f for f, d in drop_res.items() if d > P["drop_gain"]]
    summary["dropped"] = useless
    if useless:
        R.run("Градиентный бустинг (после подбора), без " + ", ".join(useless), "gb",
              lambda: make_pipeline(HistGradientBoostingClassifier(early_stopping=False, random_state=rs, **gb_best),
                                    drop=tuple(useless), scale=False), gb_best | {"drop": useless})

    exp = pd.DataFrame(R.rows)
    exp.insert(0, "date", P["run_date"])
    exp.to_csv("reports/experiments.csv", index=False)
    json.dump(summary, open(OUT / "summary.json", "w"), ensure_ascii=False, indent=1, default=str)
    # время обучения меняется от запуска к запуску, поэтому оно лежит отдельно
    # от файлов, за которыми следит DVC (иначе dvc.lock не воспроизводился бы)
    json.dump(R.timing, open(OUT / "timing.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
