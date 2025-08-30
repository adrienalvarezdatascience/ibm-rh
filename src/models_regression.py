# models_regression.py
# Régression (ex : MonthlyIncome). Version CV (OOF) :
#   - Modèles : LinReg, DecisionTree, RandomForest
#   - KPI sur OOF : MAE, R², MAPE
#   - Fairness : MAPE par groupe + ratios de parité (sur OOF)
#
# Notes :
# - On évalue chaque modèle avec cross_val_predict (5-fold CV).
# - On choisit le meilleur modèle par MAE OOF.
# - On refit ce meilleur pipeline sur TOUT le dataset pour un usage ultérieur.
# - On évite groupby.apply (FutureWarning) et on explicite observed=False.

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def _build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    """Encodage catégoriel + normalisation numérique (dense si possible)."""
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >= 1.2
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)         # compat anciennes versions
    return ColumnTransformer([
        ("cat", ohe, cat_cols),
        ("num", StandardScaler(), num_cols),
    ])


def mape_safe(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    """MAPE robuste quand y peut être 0 (on ignore les y≈0)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def train_and_compare(
    X: pd.DataFrame,
    y: np.ndarray,
    cat_cols: List[str],
    num_cols: List[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Pipeline, np.ndarray]:
    """
    Compare LinReg / Tree / RF avec des prédictions OOF (cross-validation).
    Sélectionne le meilleur modèle par MAE OOF.

    Retourne :
      - scores (DataFrame) : MAE/R²/MAPE basés sur OOF
      - best_pipe (Pipeline) : pipeline du meilleur modèle, entraîné sur TOUT X,y
      - yhat_oof_best (np.ndarray) : prédictions OOF du meilleur modèle (pour fairness)
    """
    prepro = _build_preprocessor(cat_cols, num_cols)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    models = {
        "LinReg": LinearRegression(),
        "Tree":   DecisionTreeRegressor(random_state=random_state),
        "RF":     RandomForestRegressor(n_estimators=400, random_state=random_state, n_jobs=-1),
    }

    rows = []
    oof_by_model: Dict[str, np.ndarray] = {}
    pipes_by_model: Dict[str, Pipeline] = {}

    # Évalue chaque modèle en OOF
    for name, reg in models.items():
        pipe = Pipeline([("prep", prepro), ("reg", reg)])
        # Prédictions hors-pli = jamais sur les données d'entraînement du pli
        yhat_oof = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1, method="predict")
        oof_by_model[name] = yhat_oof
        pipes_by_model[name] = pipe  # structure identique (sera refit ensuite)

        mae  = mean_absolute_error(y, yhat_oof)
        r2   = r2_score(y, yhat_oof)
        mape = mape_safe(y, yhat_oof) * 100.0

        rows.append({"Model": name, "MAE": round(mae, 2), "R²": round(r2, 3), "MAPE (%)": round(mape, 2)})

    scores = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

    # Meilleur modèle par MAE OOF
    best_name = scores.iloc[0]["Model"]
    yhat_oof_best = oof_by_model[best_name]

    # Refit du meilleur pipeline sur tout le dataset (pour l'usage prod/what-if)
    best_pipe = pipes_by_model[best_name]
    best_pipe.fit(X, y)

    return scores, best_pipe, yhat_oof_best


def fairness_tables(
    X: pd.DataFrame,
    y: np.ndarray,
    yhat: np.ndarray,
    groups: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Calcule la MAPE par groupe (sur OOF de préférence).
    Ex : Gender, MaritalStatus, tranches d'Age.
    """
    if groups is None:
        groups = [c for c in ["Gender", "MaritalStatus"] if c in X.columns]

    rep: Dict[str, pd.DataFrame] = {}

    # Groupes catégoriels simples
    for col in groups:
        if col not in X.columns:
            continue
        tmp = pd.DataFrame({"grp": X[col], "y": y, "yhat": yhat})
        rows = []
        for g, s in tmp.groupby("grp", observed=False):  # observed=False = comportement actuel
            rows.append({
                col: g,
                "n": int(s.shape[0]),
                "MAPE (%)": mape_safe(s["y"].values, s["yhat"].values) * 100.0
            })
        rep[col] = pd.DataFrame(rows)

    # Tranches d'âge si dispo
    if "Age" in X.columns:
        bins = (0, 29, 39, 49, 200)
        labels = ["<30", "30-39", "40-49", "50+"]
        grp = pd.cut(X["Age"], bins=bins, labels=labels, include_lowest=True, right=True)
        tmp = pd.DataFrame({"age_group": grp, "y": y, "yhat": yhat})
        rows_age = []
        for g, s in tmp.groupby("age_group", observed=False):
            rows_age.append({
                "age_group": g,
                "n": int(s.shape[0]),
                "MAPE (%)": mape_safe(s["y"].values, s["yhat"].values) * 100.0
            })
        rep["age_group"] = pd.DataFrame(rows_age)

    return rep


def parity_ratio(tab: pd.DataFrame, col_name: str, a: str, b: str) -> float:
    """
    Ratio simple de parité sur la MAPE : MAPE(a) / MAPE(b).
    Pratique pour dire : “erreur Femmes = 1.03× erreur Hommes”.
    """
    if tab is None or col_name not in tab.columns:
        return float("nan")
    va = tab.loc[tab[col_name] == a, "MAPE (%)"]
    vb = tab.loc[tab[col_name] == b, "MAPE (%)"]
    if len(va) == 0 or len(vb) == 0:
        return float("nan")
    denom = float(vb.iloc[0])
    return float(va.iloc[0]) / denom if denom != 0 else float("nan")
