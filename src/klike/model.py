"""Treino, avaliação e persistência dos modelos do klike_score.

Uso: `python -m klike.model` treina o modelo A em toda a base e salva em models/.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline

from klike.data import TARGET, load_clean
from klike.features import build_features
from klike.paths import MODELS

MODEL_PATH = MODELS / "score_model.joblib"
CV = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
SCORING = {"RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error", "R2": "r2"}


def make_linear() -> Pipeline:
    """Regressão linear com regularização leve.

    Sem padronização das features: assim cada coeficiente fica em "pontos de score por
    unidade" (por exemplo, +X pontos por ter hook), o que é a base da engenharia reversa.
    """
    return make_pipeline(SimpleImputer(strategy="median"), RidgeCV(alphas=np.logspace(-3, 2, 30)))


def make_gbm() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=20, random_state=42
    )


def candidates() -> dict[str, tuple[object, bool]]:
    """Modelos comparados: nome → (estimador, usa métricas de performance?)."""
    return {
        "Baseline (média)": (DummyRegressor(), False),
        "A · Linear": (make_linear(), False),
        "A · Gradient Boosting": (make_gbm(), False),
        "B · Linear": (make_linear(), True),
        "B · Gradient Boosting": (make_gbm(), True),
    }


def cross_validate_models(df: pd.DataFrame) -> pd.DataFrame:
    y = df[TARGET]
    rows = []
    for name, (model, perf) in candidates().items():
        X = build_features(df, include_performance=perf)
        res = cross_validate(model, X, y, cv=CV, scoring=SCORING)
        row = {"modelo": name}
        for metric in SCORING:
            vals = res[f"test_{metric}"] * (-1 if metric != "R2" else 1)
            row[metric] = vals.mean()
            row[f"{metric} (dp)"] = vals.std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("modelo")


def coefficients(model: Pipeline, feature_names: list[str]) -> pd.Series:
    """Coeficientes do modelo linear: pontos de score por unidade de cada feature."""
    return pd.Series(model[-1].coef_, index=feature_names)


def train(df: pd.DataFrame | None = None) -> Pipeline:
    """Treina o modelo A (pré-lançamento) em toda a base. É o modelo usado pela engine."""
    df = load_clean() if df is None else df
    X = build_features(df)
    return make_linear().fit(X, df[TARGET])


def save(model: Pipeline, path=MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path=MODEL_PATH) -> Pipeline:
    """Carrega o modelo salvo; se não existir, treina e salva."""
    if not path.exists():
        save(train(), path)
    return joblib.load(path)


if __name__ == "__main__":
    model = train()
    save(model)
    print(f"Modelo A salvo em {MODEL_PATH} (alpha={model[-1].alpha_:.3g})")
