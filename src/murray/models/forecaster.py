"""Barmah level forecaster: one LightGBM model per horizon (direct multi-horizon strategy).

Each model predicts the CHANGE from the last observed level (target_hH - barmah_level_m),
so "predict 0" is exactly the persistence baseline and the model only learns the deviation.
Packaged as one MLflow pyfunc: input = feature rows, output = levels h1..h7 in metres.
"""

import lightgbm as lgb
import mlflow
import pandas as pd

from murray.transforms.features import FEATURE_COLUMNS
from murray.transforms.labels import HORIZONS

DEFAULT_PARAMS = {
    "objective": "l1",          # optimise MAE directly (robust to flood outliers)
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
}


def _delta_target(df: pd.DataFrame, h: int) -> pd.Series:
    return df[f"target_h{h}"] - df["barmah_level_m"]


def train(train_df: pd.DataFrame, val_df: pd.DataFrame, params: dict | None = None,
          num_boost_round: int = 2000, early_stopping_rounds: int = 100,
          feature_columns: list[str] | None = None) -> dict[int, lgb.Booster]:
    """Fit one booster per horizon with early stopping on the validation period."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    cols = list(feature_columns or FEATURE_COLUMNS)
    boosters = {}
    for h in HORIZONS:
        tr = train_df[train_df[f"target_h{h}"].notna()]
        va = val_df[val_df[f"target_h{h}"].notna()]
        dtrain = lgb.Dataset(tr[cols], _delta_target(tr, h))
        dval = lgb.Dataset(va[cols], _delta_target(va, h), reference=dtrain)
        boosters[h] = lgb.train(
            params, dtrain, num_boost_round=num_boost_round, valid_sets=[dval],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
    return boosters


class BarmahForecaster(mlflow.pyfunc.PythonModel):
    """MLflow-loadable bundle of the per-horizon boosters (stored as text, not pickled objects)."""

    def __init__(self, boosters: dict[int, lgb.Booster], feature_columns: list[str] | None = None):
        self.model_strings = {h: b.model_to_string(num_iteration=b.best_iteration)
                              for h, b in boosters.items()}
        # Inference reads this to know which Feast features to fetch.
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        self._boosters = None

    def _load(self):
        if self._boosters is None:
            self._boosters = {h: lgb.Booster(model_str=s) for h, s in self.model_strings.items()}
        return self._boosters

    def __getstate__(self):  # don't pickle the live boosters
        state = self.__dict__.copy()
        state["_boosters"] = None
        return state

    def contributions(self, model_input: pd.DataFrame) -> dict[int, pd.DataFrame]:
        """Exact TreeSHAP per horizon, in metres of predicted CHANGE.

        Columns: one per feature plus "_base" (the model's average predicted change). For each
        row, the columns sum to that horizon's predicted change from today's level.
        """
        x = model_input[self.feature_columns].astype("float64")
        out = {}
        for h, b in sorted(self._load().items()):
            contrib = b.predict(x, pred_contrib=True)
            out[h] = pd.DataFrame(contrib, columns=[*self.feature_columns, "_base"],
                                  index=model_input.index)
        return out

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        x = model_input[self.feature_columns].astype("float64")
        current = model_input["barmah_level_m"].to_numpy(dtype="float64")
        return pd.DataFrame(
            {f"h{h}": current + b.predict(x) for h, b in sorted(self._load().items())},
            index=model_input.index,
        )


def best_iterations(boosters: dict[int, lgb.Booster]) -> dict[str, int]:
    return {f"best_iter_h{h}": int(b.best_iteration) for h, b in boosters.items()}


def feature_importance(boosters: dict[int, lgb.Booster]) -> pd.DataFrame:
    """Gain importance per feature, averaged across horizons (normalised to sum to 1)."""
    imp = pd.DataFrame({h: pd.Series(b.feature_importance("gain"), index=b.feature_name())
                        for h, b in boosters.items()})
    mean = imp.mean(axis=1)
    return (mean / mean.sum()).sort_values(ascending=False).rename("gain_share").to_frame()

