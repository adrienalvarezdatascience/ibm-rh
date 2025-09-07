# Objectifs : transformer les features, entraîner KMeans, puis produire des profils lisibles.

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def build_matrix(
    df: pd.DataFrame,
    cat_cols: List[str],
    num_cols: List[str]
) -> Tuple[np.ndarray, ColumnTransformer]:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    prepro = ColumnTransformer([
        ("cat", ohe, cat_cols),
        ("num", StandardScaler(), num_cols),
    ])
    Xmat = prepro.fit_transform(df)
    return Xmat, prepro


def fit_kmeans(Xmat: np.ndarray, k: int) -> KMeans:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(Xmat)
    return km


def score_range(Xmat: np.ndarray, k_min: int = 2, k_max: int = 10) -> pd.DataFrame:
    # Calcule silhouette et inertia pour différents K.
    # Utile pour choisir un K raisonnable
    rows = []
    for k in range(k_min, k_max + 1):
        km = fit_kmeans(Xmat, k)
        sil = silhouette_score(Xmat, km.labels_)
        rows.append({"K": k, "silhouette": sil, "inertia": km.inertia_})
    return pd.DataFrame(rows)


def profile_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    cat_cols: List[str],
    num_cols: List[str]
) -> pd.DataFrame:
    
    # Produit un tableau concis par cluster :
    dfp = df.copy()
    dfp["Cluster"] = labels
    profiles = []
    for k, sub in dfp.groupby("Cluster"):
        row = {"Cluster": int(k), "Count": int(len(sub))}
        for c in num_cols:
            row[f"mean_{c}"] = float(sub[c].mean())
        for c in cat_cols:
            vc = sub[c].value_counts(normalize=True, dropna=False)
            if not vc.empty:
                row[f"top_{c}"] = str(vc.index[0])
                row[f"top_{c}_pct"] = float(round(vc.iloc[0], 3))
        profiles.append(row)
    return pd.DataFrame(profiles).sort_values("Cluster").reset_index(drop=True)
