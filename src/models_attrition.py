# models_attrition.py
# Classification attrition : entraînement, CV PR-AUC/F1, seuil coût, prédiction

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class AttritionResults:
    pr_auc: Dict[str, float]
    f1_at_05: Dict[str, float]
    best_f1: Dict[str, float]
    best_thr: Dict[str, float]
    best_model_name: str
    best_pipeline: Pipeline


class AttritionModel:
    """
    Compare 4 modèles (LogReg, Tree, RF, KNN) avec le même préprocesseur.
    Choisit le meilleur en PR-AUC et l'entraîne sur tout le dataset.
    """

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.pipelines: Dict[str, Pipeline] = {
            "LogReg": Pipeline([("prep", self.preprocessor),
                                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
            "Tree":   Pipeline([("prep", self.preprocessor),
                                ("clf",  DecisionTreeClassifier(class_weight="balanced", random_state=42))]),
            "RF":     Pipeline([("prep", self.preprocessor),
                                ("clf",  RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                                                random_state=42, n_jobs=-1))]),
            "KNN":    Pipeline([("prep", self.preprocessor),
                                ("clf",  KNeighborsClassifier(n_neighbors=25))]),
        }
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.best_pipeline: Pipeline = None
        self.best_name: str = None

    def _evaluate_one(self, name: str, X: pd.DataFrame, y: np.ndarray):
        pipe = self.pipelines[name]
        proba = cross_val_predict(pipe, X, y, cv=self.cv, method="predict_proba", n_jobs=-1)[:, 1]
        pr_auc = average_precision_score(y, proba)
        f1_05  = f1_score(y, (proba >= 0.5).astype(int))
        p, r, thr = precision_recall_curve(y, proba)
        f1s = (2 * p * r / (p + r + 1e-12))
        idx = int(np.nanargmax(f1s))
        best_f1 = float(f1s[idx])
        best_thr = 0.0 if idx == 0 else float(thr[idx - 1])
        return pr_auc, f1_05, best_f1, best_thr, proba

    def fit_and_select(self, X: pd.DataFrame, y: np.ndarray) -> AttritionResults:
        pr_auc, f1_05, best_f1, best_thr = {}, {}, {}, {}
        probas_cache: Dict[str, np.ndarray] = {}

        for name in self.pipelines:
            ap, f1v, bf1, bthr, proba = self._evaluate_one(name, X, y)
            pr_auc[name], f1_05[name], best_f1[name], best_thr[name] = ap, f1v, bf1, bthr
            probas_cache[name] = proba

        self.best_name = max(pr_auc, key=pr_auc.get)
        self.best_pipeline = self.pipelines[self.best_name]
        self.best_pipeline.fit(X, y)

        return AttritionResults(
            pr_auc=pr_auc,
            f1_at_05=f1_05,
            best_f1=best_f1,
            best_thr=best_thr,
            best_model_name=self.best_name,
            best_pipeline=self.best_pipeline
        )

    @staticmethod
    def cost_optimal_threshold(y_true: np.ndarray, proba: np.ndarray, c_fn: float = 10.0, c_fp: float = 1.0, n: int = 201) -> Tuple[float, float]:
        """
        Cherche le seuil qui minimise un coût simple : C_fn * FN + C_fp * FP.
        """
        thresholds = np.linspace(0, 1, n)
        best_t, best_cost = 0.5, np.inf
        for t in thresholds:
            pred = (proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            cost = c_fn * fn + c_fp * fp
            if cost < best_cost:
                best_cost, best_t = cost, t
        return float(best_t), float(best_cost)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.best_pipeline.predict_proba(X)[:, 1]
