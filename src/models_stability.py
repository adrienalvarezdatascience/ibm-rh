# models_stability.py
# Modélisation de la stabilité (YearsAtCompany) + top facteurs via SHAP (agrégé par variable d’origine).
# Pas de graphiques ici : on renvoie juste des tableaux faciles à exploiter ailleurs.

from typing import List, Tuple
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import shap


def _build_preprocessor_dense(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    """Préprocesseur dense (important pour SHAP + RandomForest)."""
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("cat", ohe, cat_cols),
        ("num", StandardScaler(), num_cols),
    ])


def train_regression(
    X: pd.DataFrame,
    y: np.ndarray,
    cat_cols: List[str],
    num_cols: List[str],
) -> Tuple[pd.DataFrame, Pipeline, np.ndarray]:
    """
    Modèle simple et robuste : RandomForestRegressor (souvent très performant ici).
    Retourne :
      - scores (MAE, R²)
      - pipeline entraîné
      - prédictions complètes
    """
    prepro = _build_preprocessor_dense(cat_cols, num_cols)
    reg = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    pipe = Pipeline([("prep", prepro), ("reg", reg)])

    pipe.fit(X, y)
    yhat = pipe.predict(X)

    mae = mean_absolute_error(y, yhat)
    r2 = r2_score(y, yhat)
    scores = pd.DataFrame([{"Model": "RF", "MAE": round(mae, 2), "R²": round(r2, 3)}])

    return scores, pipe, yhat


def _shap_values_2d(shap_out, is_classifier: bool, class_index: int = 1) -> np.ndarray:
    """
    Harmonise la sortie SHAP vers un array 2D (n_samples, n_features),
    peu importe la version de SHAP.
    """
    if isinstance(shap_out, list):
        # SHAP renvoie une liste pour les classifieurs (par classe)
        arr = np.asarray(shap_out[class_index] if is_classifier else shap_out[0])
    else:
        arr = np.asarray(getattr(shap_out, "values", shap_out))
        if arr.ndim == 3:
            arr = arr[:, :, class_index] if is_classifier else arr.mean(axis=-1)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def top_factors_shap_aggregated(
    X: pd.DataFrame,
    y: np.ndarray,
    cat_cols: List[str],
    num_cols: List[str],
    task: str = "reg",        # "reg" (stabilité) ou "clf" (attrition)
    topn: int = 10,
) -> pd.DataFrame:
    """
    Entraîne un petit RandomForest (dense), calcule SHAP,
    puis agrège les importances par variable d’origine (pas par one-hot).
    Retourne un DataFrame (base_feat, abs_shap agrégé).
    """
    prepro = _build_preprocessor_dense(cat_cols, num_cols)
    Xd = prepro.fit_transform(X)

    if task == "reg":
        model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1).fit(Xd, y)
        is_clf = False
        class_index = 0
    else:
        model = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1).fit(Xd, y)
        is_clf = True
        class_index = 1

    # Noms de colonnes transformées
    cat_names = prepro.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
    feat_names = cat_names + num_cols

    # Mapping "feature transformée" -> "colonne d'origine"
    base_idx: List[str] = []
    for col, cats in zip(cat_cols, prepro.named_transformers_["cat"].categories_):
        base_idx.extend([col] * len(cats))
    base_idx.extend(num_cols)

    # SHAP sur un sous-échantillon (rapide et suffisant)
    rng = np.random.RandomState(42)
    idx = rng.choice(Xd.shape[0], size=min(500, Xd.shape[0]), replace=False)
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(Xd[idx])
    sv = _shap_values_2d(raw, is_classifier=is_clf, class_index=class_index)

    mean_abs = np.abs(sv).mean(axis=0).ravel()
    imp = pd.DataFrame({
        "feat_full": feat_names,
        "base_feat": base_idx,
        "abs_shap": mean_abs
    })
    agg = (imp.groupby("base_feat", as_index=False)["abs_shap"]
           .sum()
           .sort_values("abs_shap", ascending=False)
           .head(topn)
           .reset_index(drop=True))
    return agg
