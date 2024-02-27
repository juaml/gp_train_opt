# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% Config visuals
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.5, rc={"xtick.labelsize": 12})
palette = "flare"
# %% Config paths
results_path = Path(__file__).parent / "results"
figs_path = Path(__file__).parent / "figs"
# %% Plot splits
data_kind = "S4_R4"
dfs = []
for fname in results_path.glob(f"results_{data_kind}_*_splits_cv_scores.csv"):
    t_df = pd.read_csv(fname)
    n_splits = int(fname.stem.replace(f"results_{data_kind}", "").split("_")[1])
    if n_splits == 0:
        n_splits += 1
    t_df["n_splits"] = n_splits
    dfs.append(t_df)
results_df = pd.concat(dfs)

fig, axes = plt.subplots(2, 1, figsize=(5, 10))
sns.swarmplot(
    x="n_splits",
    y="fit_time",
    data=results_df,
    hue="fold",
    ax=axes[0],
)

sns.swarmplot(
    x="n_splits",
    y="test_neg_mean_absolute_error",
    data=results_df,
    hue="fold",
    ax=axes[1],
)
fig.subplots_adjust(wspace=0.3)
if data_kind == "S4_R4":
    fig.suptitle(f"fastGPR for {data_kind} (29852 features)")
else:
    fig.suptitle(f"fastGPR for {data_kind} (PCA)")

# %% Plot n_repeats + splits

data_kind = "S4_R8"

dfs = []
for fname in results_path.glob(
    f"results_{data_kind}_*_splits_*_repeats_stratified_cv_scores.csv"
):
    t_df = pd.read_csv(fname)
    n_splits = int(fname.stem.replace(f"results_{data_kind}", "").split("_")[1])
    if n_splits == 0:
        n_splits += 1
    t_df["n_splits"] = n_splits
    dfs.append(t_df)
results_df = pd.concat(dfs)

baseline_df = pd.read_csv(results_path / f"results_{data_kind}_0_splits_cv_scores.csv")
base_stats = baseline_df[
    [
        "fit_time",
        "test_neg_mean_absolute_error",
        "test_neg_mean_squared_error",
        "test_r2",
    ]
].mean()

stats_df = results_df.groupby(["n_repeats", "n_splits"])[
    [
        "fit_time",
        "test_neg_mean_absolute_error",
        "test_neg_mean_squared_error",
        "test_r2",
    ]
].mean()
stats_df = stats_df.reset_index()
fig, axes = plt.subplots(2, 1, figsize=(5, 9))

sns.swarmplot(
    x="n_splits",
    y="test_neg_mean_absolute_error",
    data=stats_df,
    hue="n_repeats",
    ax=axes[0],
    legend="full",
    palette=palette,
)
axes[0].axhline(
    base_stats["test_neg_mean_absolute_error"],
    color="red",
    linestyle="--",
    label="baseline",
)
axes[0].legend(loc=3, prop={"size": 9}, title="# Repeats")
axes[0].set_xlabel("# Splits")
axes[0].set_ylabel("-(Mean Absolute Error)")

sns.swarmplot(
    x="n_splits",
    y="fit_time",
    data=stats_df,
    hue="n_repeats",
    ax=axes[1],
    legend=False,
    palette=palette,
)
axes[1].axhline(
    base_stats["fit_time"],
    color="red",
    linestyle="--",
    label="No Split",
)
axes[1].set_xlabel("# Splits")
axes[1].set_ylabel("Fit time (s)")

fig.suptitle(f"fastGPR performance\nby number of splits and repeats")
plt.savefig(figs_path / f"fig_{data_kind}_repeats_splits.pdf", bbox_inches="tight")
fig.suptitle(None)
plt.savefig(figs_path / f"fig_{data_kind}_repeats_splits_paper.pdf", bbox_inches="tight")
stats_df.to_csv(figs_path / f"results_{data_kind}_repeats_splits.csv")
base_stats.to_csv(figs_path / f"results_{data_kind}_GPR.csv")

# %% Plot n_features + splits

data_kind = "S4_R8"

dfs = []
for fname in results_path.glob(
    f"results_{data_kind}_*_splits_*_features_stratified_cv_scores.csv"
):
    t_df = pd.read_csv(fname)
    n_splits = int(fname.stem.replace(f"results_{data_kind}", "").split("_")[1])
    if n_splits == 0:
        n_splits += 1
    t_df["n_splits"] = n_splits
    dfs.append(t_df)
results_df = pd.concat(dfs)


stats_df = results_df.groupby(["n_features", "n_splits"])[
    ["fit_time", "test_neg_mean_absolute_error"]
].mean()
stats_df = stats_df.reset_index()
fig, axes = plt.subplots(2, 1, figsize=(5, 9))

sns.swarmplot(
    x="n_splits",
    y="test_neg_mean_absolute_error",
    data=stats_df,
    hue="n_features",
    palette=palette,
    ax=axes[0],
)

axes[0].legend(loc=1, prop={"size": 9}, title="# Features")
axes[0].set_xlabel("# Splits")
axes[0].set_ylabel("-(Mean Absolute Error)")
axes[0].set_xticklabels(
    ["No split", "2 (2-rep)", "3 (2-rep)", "4 (2-rep)", "5 (2-rep)"]
)

sns.swarmplot(
    x="n_splits",
    y="fit_time",
    data=stats_df,
    hue="n_features",
    ax=axes[1],
    palette=palette,
    legend=False,
)
axes[1].set_xlabel("# Splits")
axes[1].set_ylabel("Fit time (s)")
axes[1].set_xticklabels(
    ["No split", "2 (2-rep)", "3 (2-rep)", "4 (2-rep)", "5 (2-rep)"]
)

# fig.subplots_adjust(wspace=0.2)

fig.suptitle("fastGPR performance\nby number of features and splits")
plt.savefig(figs_path / f"fig_{data_kind}_features_splits.pdf", bbox_inches="tight")
fig.suptitle(None)
plt.savefig(figs_path / f"fig_{data_kind}_features_splits_paper.pdf", bbox_inches="tight")
stats_df.to_csv(figs_path / f"results_{data_kind}_features_splits.csv")

# %%
