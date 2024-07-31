import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("medical_examination.csv")

# 2
df["overweight"] = df["weight"] / ((df["height"] / 100) ** 2)
df["overweight"] = df["overweight"].apply(lambda x: 1 if x > 25 else 0)

# 3
df["cholesterol"] = df["cholesterol"].apply(lambda x: 0 if x == 1 else 1)
df["gluc"] = df["gluc"].apply(lambda x: 0 if x == 1 else 1)


# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"],
    )

    # 6
    df_cat["total"] = 1
    df_cat = (
        df_cat.groupby(["cardio", "variable", "value"])["variable"]
        .count()
        .reset_index(name="total")
    )

    # 7
    graph = sns.catplot(
        data=df_cat, x="variable", y="total", hue="value", kind="bar", col="cardio"
    ).figure

    # 8
    fig = graph

    # 9
    fig.savefig("catplot.png")
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"])
        & (df["height"] >= df["height"].quantile(0.025))
        & (df["height"] <= df["height"].quantile(0.975))
        & (df["weight"] >= df["weight"].quantile(0.025))
        & (df["weight"] <= df["weight"].quantile(0.975))
    ]

    # 12
    corr = df_heat.corr(method="pearson")

    # 13
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    # 14
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15
    sns.heatmap(
        corr,
        vmin=-0.08,
        vmax=0.24,
        center=0.0,
        annot=True,
        fmt=".1f",
        linewidths=1,
        square=True,
        mask=mask,
        cbar_kws={"shrink": 0.5},
    )
    # 16
    fig.savefig("heatmap.png")
    return fig
