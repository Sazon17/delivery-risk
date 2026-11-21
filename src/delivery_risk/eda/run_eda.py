"""Этап eda: разведочный анализ исходной части.

Все значения, на которые опирается отчёт ПР2, пишутся в
reports/eda_summary.json, рисунки - в reports/figures.
"""
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # запуск файлом: python src/delivery_risk/...
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml
from scipy import stats

from delivery_risk.data.geo import distance_km, load_reference
from delivery_risk.data.history import load_events, seller_history
from delivery_risk.data.io import read_labels, read_orders
from delivery_risk.data.validate import check_orders

FIG = Path("reports/figures")
plt.rcParams.update({"font.family": "serif", "font.size": 9})


def pct(x):
    return round(100 * float(x), 2)


def odds_ratio(model, term, scale=1.0):
    b = model.params[term] * scale
    lo, hi = model.conf_int().loc[term] * scale
    return {"or": round(float(np.exp(b)), 2), "ci": [round(float(np.exp(lo)), 2),
            round(float(np.exp(hi)), 2)], "p": round(float(model.pvalues[term]), 3)}


def logit(formula, data):
    return smf.logit(formula, data=data).fit(disp=0, maxiter=200)


def frame():
    q = yaml.safe_load(open("configs/params.yaml"))["quality"]
    o = read_orders("data/interim/initial.csv")
    clean, _ = check_orders(o, q)
    lab = read_labels("data/interim/labels_00.csv")
    df = clean.merge(lab, on="order_id", how="left")
    df["dist_km"], _ = distance_km(df, load_reference())
    df["promise_days"] = (df["order_estimated_delivery_date"].dt.normalize()
                          - df["order_purchase_timestamp"].dt.normalize()).dt.days
    df["approval_h"] = (df["order_approved_at"] - df["order_purchase_timestamp"]).dt.total_seconds() / 3600
    df["month"] = df["order_purchase_timestamp"].dt.to_period("M")
    day = df["order_purchase_timestamp"].dt.normalize()
    daily = day.value_counts().sort_index()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()), fill_value=0)
    prev7 = daily.rolling(7, min_periods=1).sum().shift(1).fillna(0)
    df["load_7d"] = day.map(prev7) / 1000
    h = seller_history(df, load_events())
    df = pd.concat([df, h], axis=1)
    return df


def main():
    df = frame()
    lab = df[df["is_late"].notna()].copy()
    lab["is_late"] = lab["is_late"].astype(int)
    S = {"orders": int(len(df)), "labelled": int(len(lab)), "late": int(lab["is_late"].sum()),
         "late_rate": pct(lab["is_late"].mean())}
    S["status_unlabelled"] = df.loc[df["is_late"].isna(), "order_status"].value_counts().to_dict()

    # состав строки заказа (Таблица 1.2 ПР2)
    S["approval_median_min"] = round(float(df["approval_h"].median() * 60), 1)
    S["promise_days"] = {"min": int(df["promise_days"].min()), "median": float(df["promise_days"].median()),
                         "max": int(df["promise_days"].max())}
    S["share_customer_sp"] = pct((df["customer_state"] == "SP").mean())
    S["share_interstate"] = pct((df["customer_state"] != df["seller_state"]).mean())
    S["share_single_item"] = pct((df["n_items"] == 1).mean())
    S["items_price_median"] = float(df["items_price"].median())
    S["sellers"] = int(df["seller_id"].nunique())
    S["categories"] = int(df["product_category_name"].nunique())
    S["payment_share"] = {k: pct(v) for k, v in df["payment_type"].value_counts(normalize=True).items()}
    rep = df.groupby("customer_unique_id")["order_id"].transform("size") > 1
    S["repeat_buyer_orders"] = pct(rep.mean())

    # доля опозданий и обещанный срок по месяцам (Рисунок 1.1 ПР2)
    m = lab.groupby("month").agg(rate=("is_late", "mean"), promise=("promise_days", "median"))
    m = m[m.index >= pd.Period("2017-01", "M")]
    S["monthly"] = {str(k): [pct(r.rate), float(r.promise)] for k, r in m.iterrows()}
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(6.2, 3.6), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    x = np.arange(len(m))
    ax[0].bar(x, m["rate"] * 100, color="0.6", edgecolor="0.2")
    ax[0].set_ylabel("опоздания, %")
    ax[1].plot(x, m["promise"], "k-o", ms=3)
    ax[1].set_ylabel("медиана срока, дн.")
    ax[1].set_xticks(x, [p.strftime("%m.%y") for p in m.index], rotation=90)
    fig.tight_layout()
    fig.savefig(FIG / "late_by_month.png", dpi=200)
    plt.close(fig)

    # числовые признаки у опоздавших и вовремя (Манн - Уитни)
    mw = {}
    for c in ["dist_km", "promise_days", "items_freight", "weight_g_total", "items_price"]:
        a, b = lab.loc[lab["is_late"] == 1, c].dropna(), lab.loc[lab["is_late"] == 0, c].dropna()
        mw[c] = {"late_median": round(float(a.median()), 1), "ontime_median": round(float(b.median()), 1),
                 "p": float(stats.mannwhitneyu(a, b).pvalue)}
    S["mann_whitney"] = mw

    # срезы (Рисунок 1.3 ПР2)
    lab["dist_bin"] = pd.cut(lab["dist_km"], [-1, 100, 500, 1000, 2000, 1e5],
                             labels=["<=100", "100-500", "500-1000", "1000-2000", ">2000"])
    lab["appr_bin"] = pd.cut(lab["approval_h"], [-1, 1, 24, 48, 1e5], labels=["<=1", "1-24", "24-48", ">48"])
    lab["load_q"] = pd.qcut(lab["load_7d"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    S["slices"] = {
        "distance": {str(k): pct(v) for k, v in lab.groupby("dist_bin", observed=True)["is_late"].mean().items()},
        "customer_state": {k: pct(v) for k, v in lab.groupby("customer_state")["is_late"].mean().items()},
        "approval": {str(k): pct(v) for k, v in lab.groupby("appr_bin", observed=True)["is_late"].mean().items()},
        "load_quartile": {str(k): pct(v) for k, v in lab.groupby("load_q", observed=True)["is_late"].mean().items()},
    }
    dow = pd.crosstab(lab["order_purchase_timestamp"].dt.dayofweek, lab["is_late"])
    S["weekday_chi2_p"] = round(float(stats.chi2_contingency(dow)[1]), 2)

    # ранговые корреляции (Рисунок 1.4 ПР2)
    num = ["dist_km", "promise_days", "items_freight", "items_price", "weight_g_total",
           "volume_cm3_total", "approval_h", "is_late"]
    sp = lab[num].corr(method="spearman").round(2)
    S["spearman"] = sp.to_dict()

    # гипотезы Г1-Г6 (Таблица 1.4 ПР2)
    lab["interstate"] = (lab["customer_state"] != lab["seller_state"]).astype(int)
    lab["log_dist"] = np.log1p(lab["dist_km"])
    lab["promise_w"] = lab["promise_days"] / 7
    med = lab.groupby(["customer_state", "month"])["promise_days"].transform("median")
    lab["short_promise"] = (lab["promise_days"] < med).astype(int)
    hist_ok = lab["seller_n_matured"] >= 5
    rate = lab["seller_n_late"] / lab["seller_n_matured"].clip(lower=1)
    lab["seller_group"] = np.where(~hist_ok, "none", np.where(rate > 0.10, "bad", "good"))
    lab["slow_approval"] = (lab["approval_h"] > 24).astype(int)
    lab["heavy"] = (lab["weight_g_total"] > 5000).astype(int)
    lab["t"] = (lab["order_purchase_timestamp"] - lab["order_purchase_timestamp"].min()).dt.days / 30
    lab["month_s"] = lab["month"].astype(str)
    ctrl = "log_dist + promise_w + C(customer_state) + C(month_s)"   # направление, срок, сезон
    H = {}
    g = lab.groupby("interstate")["is_late"].mean()
    H["G1"] = {"rates": [pct(g[1]), pct(g[0])],
               "within_state": odds_ratio(logit("is_late ~ interstate + C(customer_state) + C(month_s)", lab), "interstate"),
               "with_distance": odds_ratio(logit("is_late ~ interstate + log_dist + C(customer_state) + C(month_s)", lab),
                                           "interstate")}
    g = lab.groupby("short_promise")["is_late"].mean()
    H["G2"] = {"rates": [pct(g[1]), pct(g[0])],
               "adjusted": odds_ratio(logit("is_late ~ promise_w + log_dist + C(customer_state)", lab), "promise_w"),
               "raw": odds_ratio(logit("is_late ~ promise_w", lab), "promise_w")}
    g = lab.groupby("seller_group")["is_late"].mean()
    mod = logit("is_late ~ C(seller_group, Treatment('good')) + " + ctrl, lab)
    H["G3"] = {"rates": {k: pct(v) for k, v in g.items()},
               "bad": odds_ratio(mod, "C(seller_group, Treatment('good'))[T.bad]"),
               "none": odds_ratio(mod, "C(seller_group, Treatment('good'))[T.none]")}
    g = lab.groupby("slow_approval")["is_late"].mean()
    H["G4"] = {"rates": [pct(g[1]), pct(g[0])],
               "adjusted": odds_ratio(logit("is_late ~ slow_approval + " + ctrl, lab),
                                      "slow_approval"),
               "boleto_slow": pct(lab.loc[lab["payment_type"] == "boleto", "slow_approval"].mean()),
               "card_slow": pct(lab.loc[lab["payment_type"] == "credit_card", "slow_approval"].mean())}
    g = lab.groupby("heavy")["is_late"].mean()
    H["G5"] = {"rates": [pct(g[1]), pct(g[0])],
               "adjusted": odds_ratio(logit("is_late ~ heavy + " + ctrl, lab), "heavy")}
    H["G6"] = {"quartiles": S["slices"]["load_quartile"],
               "adjusted": odds_ratio(logit("is_late ~ load_7d + t", lab), "load_7d")}
    S["hypotheses"] = H

    Path("reports").mkdir(exist_ok=True)
    json.dump(S, open("reports/eda_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: S[k] for k in ["labelled", "late", "late_rate", "mann_whitney", "slices"]},
                     ensure_ascii=False, indent=0)[:3000])
    print(json.dumps(H, ensure_ascii=False))


if __name__ == "__main__":
    main()
