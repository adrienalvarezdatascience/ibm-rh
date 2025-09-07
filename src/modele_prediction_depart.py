# Objectifs : comparer plusieurs modèles, calculer les KPI utiles

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    f1_score, brier_score_loss, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def _build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    """Encodage des catégorielles + normalisation des numériques."""
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("cat", ohe, cat_cols),
        ("num", StandardScaler(), num_cols),
    ])


@dataclass
class AttritionResult:
    name: str
    pr_auc: float
    f1_at_05: float
    best_f1: float
    best_thr: float
    brier: float
    proba_oof: np.ndarray


def _evaluate_model(
    name: str,
    pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: StratifiedKFold
) -> AttritionResult:
    """Évalue un pipeline en CV et sort tous les KPI utiles."""
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

    pr_auc = average_precision_score(y, proba)
    f1_05 = f1_score(y, (proba >= 0.5).astype(int))

    p, r, thr = precision_recall_curve(y, proba)
    f1s = (2 * p * r) / (p + r + 1e-12)
    best_idx = int(np.nanargmax(f1s))
    best_f1 = float(f1s[best_idx])
    best_thr = 0.0 if best_idx == 0 else float(thr[best_idx - 1])

    brier = brier_score_loss(y, proba)

    return AttritionResult(
        name=name, pr_auc=pr_auc, f1_at_05=f1_05,
        best_f1=best_f1, best_thr=best_thr, brier=brier,
        proba_oof=proba
    )


def train_and_compare(
    X: pd.DataFrame,
    y: np.ndarray,
    cat_cols: List[str],
    num_cols: List[str],
    n_splits: int = 5,
) -> Tuple[Dict[str, AttritionResult], Pipeline]:
    
    # Compare LogReg / Tree / RF / KNN avec la même prépa de features.
    prepro = _build_preprocessor(cat_cols, num_cols)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = {
        "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Tree":   DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "RF":     RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1),
        "KNN":    KNeighborsClassifier(n_neighbors=25),
    }

    results: Dict[str, AttritionResult] = {}
    for name, clf in models.items():
        pipe = Pipeline([("prep", prepro), ("clf", clf)])
        results[name] = _evaluate_model(name, pipe, X, y, cv)

    # meilleur modèle par PR-AUC
    best_name = max(results.values(), key=lambda r: r.pr_auc).name
    best_pipe = Pipeline([("prep", prepro), ("clf", models[best_name])])
    best_pipe.fit(X, y)

    return results, best_pipe


def cost_optimal_threshold(
    proba: np.ndarray,
    y: np.ndarray,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    grid_size: int = 200
) -> Tuple[float, float]:
    
    # Cherche le seuil qui minimise le coût total : coût = cost_fn * FN + cost_fp * FP
    thresholds = np.linspace(0, 1, grid_size)
    best_t, best_c = 0.5, np.inf
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        cost = cost_fn * fn + cost_fp * fp
        if cost < best_c:
            best_c, best_t = cost, t
    return float(best_t), float(best_c)
