# models_regression.py
# Régression (ex : MonthlyIncome). On garde simple :
#   - LinReg, DecisionTree, RandomForest
#   - KPI : MAE, R², MAPE
#   - Fairness : MAPE par groupe + ratios de parité

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def _build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("cat", ohe, cat_cols),
        ("num", StandardScaler(), num_cols),
    ])


def mape_safe(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    """MAPE robuste quand y peut être 0."""
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
) -> Tuple[pd.DataFrame, Pipeline, np.ndarray]:
    """
    Compare LinReg / Tree / RF et choisit le meilleur (par MAE).
    Retourne :
      - tableau des scores
      - pipeline final entraîné
      - prédictions du meilleur modèle sur tout X (utile pour fairness)
    """
    prepro = _build_preprocessor(cat_cols, num_cols)
    models = {
        "LinReg": LinearRegression(),
        "Tree":   DecisionTreeRegressor(random_state=42),
        "RF":     RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
    }

    rows = []
    best_name, best_pipe, best_preds, best_mae = None, None, None, 1e18
    for name, reg in models.items():
        pipe = Pipeline([("prep", prepro), ("reg", reg)])
        pipe.fit(X, y)
        yhat = pipe.predict(X)
        mae = mean_absolute_error(y, yhat)
        r2  = r2_score(y, yhat)
        mape = mape_safe(y, yhat) * 100
        rows.append({"Model": name, "MAE": round(mae, 2), "R²": round(r2, 3), "MAPE (%)": round(mape, 2)})

        if mae < best_mae:
            best_name, best_pipe, best_preds, best_mae = name, pipe, yhat, mae

    scores = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    return scores, best_pipe, best_preds


def fairness_tables(
    X: pd.DataFrame,
    y: np.ndarray,
    yhat: np.ndarray,
    groups: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Calcule la MAPE par groupe pour quelques colonnes (ex: Gender, MaritalStatus, tranches d'Age).
    Retourne un dict {nom_groupe -> DataFrame}.
    """
    if groups is None:
        groups = [c for c in ["Gender", "MaritalStatus"] if c in X.columns]

    rep: Dict[str, pd.DataFrame] = {}

    # Groupes catégoriels simples
    for col in groups:
        if col not in X.columns:
            continue
        tmp = pd.DataFrame({"grp": X[col], "y": y, "yhat": yhat})
        tab = tmp.groupby("grp").apply(lambda s: pd.Series({
            "n": s.shape[0],
            "MAPE (%)": mape_safe(s["y"].values, s["yhat"].values) * 100
        })).reset_index().rename(columns={"grp": col})
        rep[col] = tab

    # Tranches d'âge si dispo
    if "Age" in X.columns:
        bins = (0, 29, 39, 49, 200)
        labels = ["<30", "30-39", "40-49", "50+"]
        grp = pd.cut(X["Age"], bins=bins, labels=labels, include_lowest=True, right=True)
        tmp = pd.DataFrame({"age_group": grp, "y": y, "yhat": yhat})
        tab_age = tmp.groupby("age_group").apply(lambda s: pd.Series({
            "n": s.shape[0],
            "MAPE (%)": mape_safe(s["y"].values, s["yhat"].values) * 100
        })).reset_index()
        rep["age_group"] = tab_age

    return rep


def parity_ratio(tab: pd.DataFrame, col_name: str, a: str, b: str) -> float:
    """
    Calcule un ratio simple de parité sur la MAPE : MAPE(a) / MAPE(b).
    Pratique pour dire : “erreur Femmes = 1.03× erreur Hommes”.
    """
    va = tab.loc[tab[col_name] == a, "MAPE (%)"]
    vb = tab.loc[tab[col_name] == b, "MAPE (%)"]
    if len(va) == 0 or len(vb) == 0:
        return float("nan")
    return float(va.values[0] / vb.values[0]) if vb.values[0] != 0 else float("nan")
