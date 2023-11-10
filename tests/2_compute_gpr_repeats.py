# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.gaussian_process.kernels import RBF
from pathlib import Path
import sys
from julearn.utils import configure_logging
from julearn import run_cross_validation
from julearn.pipeline import PipelineCreator
from julearn.config import set_config
import pickle

sys.path.append(str(Path(__file__).parent.parent / "src"))
from gpr_fast import fastGaussianProcessRegressor

configure_logging(level="INFO")

# Too many features in S4_R4
set_config("disable_xtypes_check", True)
set_config("disable_x_check", True)
set_config("disable_x_verbose", True)
set_config("disable_xtypes_verbose", True)

# %% configure script
data_kind = "S4_R8"

# %% Load data
input_fname = Path(__file__).parent.parent / "data" / f"ixi_camcan_enki_{data_kind}.csv"
data_df = pd.read_csv(input_fname)
data_df["site"] = data_df.site.map(lambda x: x.split("/")[0])
# %% Set-up julearn

X = [x for x in data_df.columns if x.startswith("f_")]
y = "age"
groups = "site"

scoring = [
    "neg_mean_absolute_error",
    "neg_mean_squared_error",
    "r2",
]

cv = LeaveOneGroupOut()

for n_splits in [2, 3, 4, 5]:
    for n_repeats in range(1, 11):
        creator = PipelineCreator(problem_type="regression")
        creator.add("select_variance", threshold=1e-5)
        creator.add("pca")
        if n_splits == 0:
            creator.add(
                "gauss",
                kernel=RBF(10.0, (1e-7, 10e7)),
                normalize_y=True,
                n_restarts_optimizer=100,
                random_state=42,
            )
        else:
            creator.add(
                fastGaussianProcessRegressor(
                    kernel=RBF(10.0, (1e-7, 10e7)),
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    normalize_y=True,
                    n_restarts_optimizer=100,
                    random_state=42,
                ),
                name="fastgauss",
            )

        # %% Do CV
        scores, final_model, inspector = run_cross_validation(
            X=["f_.*"],
            X_types={"continuous": ["f_.*"]},
            y="age",
            data=data_df,
            model=creator,
            cv=cv,
            groups=groups,
            return_estimator="all",
            seed=42,
            scoring=scoring,
            return_inspector=True,
        )
        scores["model"] = "gauss"
        scores["n_repeats"] = n_repeats

        out_prefix = f"results_{data_kind}_{n_splits}_splits_{n_repeats}_repeats"
        out_path = Path(__file__).parent / "results"

        scores.to_csv(out_path / f"{out_prefix}_cv_scores.csv", index=False)

        with open(out_path / f"{out_prefix}_inspector.pkl", "wb") as f:
            pickle.dump(inspector, f)

        with open(out_path / f"{out_prefix}_final_model.pkl", "wb") as f:
            pickle.dump(final_model, f)

# %%
