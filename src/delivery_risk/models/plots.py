"""Этап plots: графики ПР3 по reports/experiments.csv и reports/pr3/summary.json.

Размер холста задан в натуральную величину под ширину страницы отчёта
(15-16 см), поэтому кегль на рисунке совпадает с печатным.
"""
import json
from pathlib import Path

import logging
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("reports/pr3")
CM = 1 / 2.54
plt.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                     "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "svg.hashsalt": "pr3", "axes.spines.top": False, "axes.spines.right": False})


def comma(fig):
    """Десятичная запятая на осях, как в тексте отчёта."""
    from matplotlib.ticker import FuncFormatter, ScalarFormatter
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if axis.get_scale() == "linear" and isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))


def save(fig, name):
    comma(fig)
    fig.savefig(OUT / name, dpi=220, metadata={"Software": None})
    plt.close(fig)


def regularization(S):
    p = pd.DataFrame(S["lr_path"])
    fig, ax = plt.subplots(1, 2, figsize=(16 * CM, 6.2 * CM))
    for pen, st in (("l2", "o-"), ("l1", "s--")):
        q = p[p["penalty"] == pen]
        ax[0].plot(q["C"], q["lift"], st, color="black", ms=4, label=pen.upper())
        ax[1].plot(q["C"], q["nonzero"], st, color="black", ms=4, label=pen.upper())
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("C (меньше – сильнее регуляризация)")
        a.grid(alpha=0.3)
        a.legend(frameon=False)
    ax[0].set_ylabel("PR-AUC / доля опозданий")
    ax[1].set_ylabel("ненулевых коэффициентов")
    ax[1].axhline(p["n_coef"].iloc[0], color="0.5", lw=0.8, ls=":")
    fig.tight_layout()
    save(fig, "regularization.png")


def tree_depth(S):
    d = pd.DataFrame(S["tree_depth"])
    labels = [str(x) for x in d["depth"]]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(15 * CM, 6.2 * CM))
    ax.plot(x, d["train_lift"], "o-", color="black", ms=4, label="обучающие блоки")
    ax.plot(x, d["valid_lift"], "s--", color="0.45", ms=4, label="проверочные месяцы")
    ax.axhline(1, color="0.6", lw=0.8, ls=":")
    ax.set_xticks(x, labels)
    ax.set_xlabel("максимальная глубина дерева")
    ax.set_ylabel("PR-AUC / доля опозданий")
    ax.set_yscale("log")
    ax.set_yticks([1, 1.5, 2, 3, 5, 10], ["1", "1,5", "2", "3", "5", "10"])
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "tree_depth.png")


def comparison(E, short):
    E = E[E["name"].isin(short)].copy()
    E["label"] = E["name"].map(short)
    E = E.sort_values("lift")
    fig, ax = plt.subplots(figsize=(16 * CM, 10.5 * CM))
    y = np.arange(len(E))
    ax.barh(y, E["lift"], xerr=E["lift_std"], color="0.7", edgecolor="0.2",
            error_kw={"ecolor": "0.3", "capsize": 2, "lw": 0.8})
    for yi, v in zip(y, E["lift"]):
        ax.text(0.05, yi, f"{v:.2f}".replace(".", ","), va="center", fontsize=8.5)
    ax.set_yticks(y, E["label"])
    ax.axvline(1, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel("lift: среднее и стандартное отклонение по пяти месяцам")
    fig.tight_layout()
    save(fig, "comparison.png")


def folds(E, S, short):
    months = ["09.17", "10.17", "11.17", "12.17", "01.18"]
    fig, ax = plt.subplots(figsize=(15 * CM, 6.4 * CM))
    styles = ["o-", "s--", "^-.", "d:"]
    for st, (name, lab) in zip(styles, short.items()):
        r = E[E["name"] == name].iloc[0]
        v = [float(t) for t in r["lift_folds"].split()]
        ax.plot(months, v, st, color="black" if st != "d:" else "0.45", ms=4, label=lab)
    rates = [f["late"] * 100 for f in S["folds"]]
    for i, r in enumerate(rates):
        ax.text(i, 0.55, f"{r:.1f} %".replace(".", ","), ha="center", fontsize=8.5, color="0.3")
    ax.axhline(1, color="0.6", lw=0.8, ls=":")
    ax.set_ylim(0.4, None)
    ax.set_xlabel("проверочный месяц (внизу – доля опозданий в нём)")
    ax.set_ylabel("PR-AUC / доля опозданий")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "folds.png")


def main():
    S = json.load(open(OUT / "summary.json"))
    E = pd.read_csv("reports/experiments.csv")
    regularization(S)
    tree_depth(S)
    names = list(E["name"])
    pick = {
        "Константа (доля опозданий)": "Константа",
        "Логистическая регрессия L2, C=1, без новых признаков": "Логрегрессия без новых признаков",
        "Градиентный бустинг, без новых признаков": "Бустинг без новых признаков",
        "Правило: доля опозданий штата": "Правило по штату",
        "k ближайших соседей, k=50": "k ближайших соседей",
        "Наивный байесовский (гауссовский)": "Наивный Байес",
        "Дерево решений без ограничения глубины": "Дерево без ограничения",
        "Случайный лес, 300 деревьев": "Случайный лес",
        "Градиентный бустинг (по умолчанию)": "Бустинг по умолчанию",
        "Градиентный бустинг (после подбора)": "Бустинг после подбора",
        "Логистическая регрессия L2 (после подбора)": "Логрегрессия после подбора",
    }
    for n in names:
        if n.startswith("Дерево решений, глубина"):
            pick[n] = "Дерево, " + n.split(", ")[1]
        if n.startswith("Градиентный бустинг (после подбора), без"):
            pick[n] = "Бустинг, отбор признаков"
        if n.startswith("Логистическая регрессия L2 (после подбора), без"):
            pick[n] = "Логрегрессия, отбор признаков"
    comparison(E, pick)
    final = json.load(open("models/model_v1.json"))["experiment"]
    lines = {final: "итоговая модель",
             "Логистическая регрессия L2 (после подбора)": "логрегрессия",
             "Случайный лес, 300 деревьев": "случайный лес",
             "Правило: доля опозданий штата": "правило по штату"}
    folds(E, S, lines)


if __name__ == "__main__":
    main()
