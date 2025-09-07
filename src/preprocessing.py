# Objectif : Fonctions de préparation des données pour le projet RH (Attrition, Clustering, Régression, Stabilité)

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def load_data(path: str) -> pd.DataFrame:
    """
    Charge le dataset IBM HR (ou autre).
    Supprime les colonnes qui ne servent jamais (identifiants, constantes).
    """
    df = pd.read_csv(path)

    # On remplace la cible "Attrition" par 0/1 si elle existe
    if "Attrition" in df.columns and df["Attrition"].dtype == object:
        df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    # Colonnes inutiles (identifiants ou constantes)
    drop_cols = [c for c in ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Crée un transformateur scikit-learn qui encode les variables catégorielles
    et normalise les variables numériques.
    """
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    num_cols = [c for c in df.columns if c not in cat_cols]

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >= 1.2
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)        # compat versions plus anciennes

    prepro = ColumnTransformer([
        ("cat", ohe, cat_cols),
        ("num", StandardScaler(), num_cols)
    ])

    return prepro, cat_cols, num_cols
