# main.py
# Script d'orchestration du projet RH (Attrition, Clustering, Régression, Stabilité)
# - Charge les données (et les nettoie)
# - Lance chaque bloc (avec nos modules src/)
# - Sauvegarde des résultats simples en CSV dans reports/

import os
import json
import urllib.request
import numpy as np
import pandas as pd

# Modules maison (src/)
from preprocessing import load_data
from models_attrition import train_and_compare as attrition_train_compare, cost_optimal_threshold
from models_clustering import build_matrix, fit_kmeans, profile_clusters, score_range
from models_regression import train_and_compare as reg_train_compare, fairness_tables, parity_ratio
from models_stability import train_regression as stab_train_regression, top_factors_shap_aggregated

# ---------------------------------------------------------------------
# 0) Chemins & setup
# ---------------------------------------------------------------------
DATA_DIR = "data"
REPORTS_DIR = os.path.join("reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")  # en cas d'usage ultérieur

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

CSV_NAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
CSV_PATH = os.path.join(DATA_DIR, CSV_NAME)
CSV_URL = "https://raw.githubusercontent.com/insaid2018/Term-2/master/Projects/Attrition/WA_Fn-UseC_-HR-Employee-Attrition.csv"

np.random.seed(42)

# Téléchargement si besoin (pratique pour rendre le projet autonome)
if not os.path.exists(CSV_PATH):
    print(f"Téléchargement du dataset vers {CSV_PATH} ...")
    urllib.request.urlretrieve(CSV_URL, CSV_PATH)

# ---------------------------------------------------------------------
# 1) Chargement & préparation de base
# ---------------------------------------------------------------------
df = load_data(CSV_PATH)  # supprime colonnes inutiles + map Attrition Yes/No -> 1/0 si présent

# Raccourcis : colonnes cat/num (réutilisés partout)
def split_cols(df_):
    cat = [c for c in df_.columns if df_[c].dtype == "object"]
    num = [c for c in df_.columns if c not in cat]
    return cat, num

print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes.")

# ---------------------------------------------------------------------
# 2) ATTRITION (classification)
# ---------------------------------------------------------------------
if "Attrition" in df.columns:
    print("\n[ATTRITION] Entraînement et comparaison de modèles...")
    y_attr = df["Attrition"].values
    X_attr = df.drop(columns=["Attrition"])
    cat_cols, num_cols = split_cols(X_attr)

    # Compare LogReg / Tree / RF / KNN et entraîne le meilleur
    results, best_pipe_attr = attrition_train_compare(X_attr, y_attr, cat_cols, num_cols)

    # Résumé des scores (PR-AUC, F1, etc.)
    attr_rows = []
    for name, res in results.items():
        attr_rows.append({
            "Model": name,
            "PR_AUC": round(res.pr_auc, 3),
            "F1@0.5": round(res.f1_at_05, 3),
            "Best_F1": round(res.best_f1, 3),
            "Best_thr": round(res.best_thr, 3),
            "Brier": round(res.brier, 3),
        })
    attr_scores_df = pd.DataFrame(attr_rows).sort_values("PR_AUC", ascending=False)
    attr_scores_df.to_csv(os.path.join(REPORTS_DIR, "attrition_metrics.csv"), index=False)

    # Seuil métier optimal (coût FN >> FP)
    best_name = attr_scores_df.iloc[0]["Model"]
    proba_oof = results[best_name].proba_oof
    thr_opt, cost_min = cost_optimal_threshold(proba_oof, y_attr, cost_fn=10.0, cost_fp=1.0)
    with open(os.path.join(REPORTS_DIR, "attrition_threshold.json"), "w") as f:
        json.dump({"best_model": best_name, "threshold_optimal": thr_opt, "cost_min": cost_min}, f, indent=2)

    print("  -> attrition_metrics.csv + attrition_threshold.json sauvegardés.")

else:
    print("\n[ATTRITION] Colonne 'Attrition' absente : bloc ignoré.")

# ---------------------------------------------------------------------
# 3) CLUSTERING (KMeans + profils)
# ---------------------------------------------------------------------
print("\n[CLUSTERING] KMeans + profils de clusters...")
df_clust = df.drop(columns=["Attrition"]) if "Attrition" in df.columns else df.copy()
cat_cols_c, num_cols_c = split_cols(df_clust)

# Matrice dense pour KMeans
Xmat, prepro_clust = build_matrix(df_clust, cat_cols_c, num_cols_c)

# Evaluation rapide des K (2..8) : silhouette + inertia
scores_k = score_range(Xmat, k_min=2, k_max=8)
scores_k.to_csv(os.path.join(REPORTS_DIR, "clustering_k_scores.csv"), index=False)

# Par défaut, on livre K=5 (bon compromis lisible pour le métier)
km5 = fit_kmeans(Xmat, k=5)
profiles_k5 = profile_clusters(df_clust, km5.labels_, cat_cols_c, num_cols_c)
profiles_k5.to_csv(os.path.join(REPORTS_DIR, "clustering_k5_profiles.csv"), index=False)

print("  -> clustering_k_scores.csv + clustering_k5_profiles.csv sauvegardés.")

# ---------------------------------------------------------------------
# 4) RÉGRESSION (MonthlyIncome) + fairness
# ---------------------------------------------------------------------
if "MonthlyIncome" in df.columns:
    print("\n[RÉGRESSION] MonthlyIncome + fairness...")
    y_mi = df["MonthlyIncome"].values
    X_mi = df.drop(columns=["MonthlyIncome"])
    cat_cols_m, num_cols_m = split_cols(X_mi)

    # Compare LinReg / Tree / RF (retourne le meilleur pipeline et ses prédictions)
    reg_scores, reg_pipe, yhat_mi = reg_train_compare(X_mi, y_mi, cat_cols_m, num_cols_m)
    reg_scores.to_csv(os.path.join(REPORTS_DIR, "regression_monthlyincome_scores.csv"), index=False)

    # Fairness (MAPE par groupes) + ratios simples
    fair_tabs = fairness_tables(X_mi, y_mi, yhat_mi, groups=["Gender", "MaritalStatus"])
    for k, tab in fair_tabs.items():
        tab.to_csv(os.path.join(REPORTS_DIR, f"regression_fairness_{k}.csv"), index=False)

    # Exemple de parité relative Femmes/Hommes si dispo
    ratios = {}
    if "Gender" in fair_tabs:
        ratios["MAPE_Female_over_Male"] = parity_ratio(fair_tabs["Gender"], "Gender", "Female", "Male")
    if "MaritalStatus" in fair_tabs:
        ratios["MAPE_Married_over_Single"] = parity_ratio(fair_tabs["MaritalStatus"], "MaritalStatus", "Married", "Single")
    # Age group si présent
    if "age_group" in fair_tabs:
        try:
            ag = fair_tabs["age_group"]
            a = float(ag.loc[ag["age_group"] == "<30", "MAPE (%)"])
            b = float(ag.loc[ag["age_group"] == "40-49", "MAPE (%)"])
            ratios["MAPE_<30_over_40-49"] = a / b if b != 0 else np.nan
        except Exception:
            pass

    with open(os.path.join(REPORTS_DIR, "regression_fairness_ratios.json"), "w") as f:
        json.dump({k: (None if v is None else float(v)) for k, v in ratios.items()}, f, indent=2)

    print("  -> regression_monthlyincome_scores.csv + fairness *.csv + ratios.json sauvegardés.")

else:
    print("\n[RÉGRESSION] Colonne 'MonthlyIncome' absente : bloc ignoré.")

# ---------------------------------------------------------------------
# 5) STABILITÉ (YearsAtCompany) + SHAP (Top-10)
# ---------------------------------------------------------------------
if "YearsAtCompany" in df.columns:
    print("\n[STABILITÉ] YearsAtCompany + SHAP Top-10...")
    y_yac = df["YearsAtCompany"].values
    X_yac = df.drop(columns=["YearsAtCompany"])
    cat_cols_s, num_cols_s = split_cols(X_yac)

    # Modèle simple (RF) + scores
    stab_scores, stab_pipe, yhat_yac = stab_train_regression(X_yac, y_yac, cat_cols_s, num_cols_s)
    stab_scores.to_csv(os.path.join(REPORTS_DIR, "stability_scores.csv"), index=False)

    # Top-10 facteurs (stabilité) par SHAP agrégé
    top10_stab = top_factors_shap_aggregated(X_yac, y_yac, cat_cols_s, num_cols_s, task="reg", topn=10)
    top10_stab.to_csv(os.path.join(REPORTS_DIR, "top10_stability.csv"), index=False)

    # Top-10 facteurs (attrition) pour comparer (si attrition dispo)
    if "Attrition" in df.columns:
        X_attr2 = df.drop(columns=["Attrition"])
        y_attr2 = df["Attrition"].values
        cat_cols_a, num_cols_a = split_cols(X_attr2)
        top10_attr = top_factors_shap_aggregated(X_attr2, y_attr2, cat_cols_a, num_cols_a, task="clf", topn=10)
        top10_attr.to_csv(os.path.join(REPORTS_DIR, "top10_attrition.csv"), index=False)

        # Intersection (leviers communs)
        common = sorted(list(set(top10_stab["base_feat"]).intersection(set(top10_attr["base_feat"]))))
        with open(os.path.join(REPORTS_DIR, "top10_common_stability_attrition.json"), "w") as f:
            json.dump({"common_variables": common}, f, indent=2)
    print("  -> stability_scores.csv + top10_stability.csv (+ top10_attrition.csv si dispo) sauvegardés.")
else:
    print("\n[STABILITÉ] Colonne 'YearsAtCompany' absente : bloc ignoré.")

# ---------------------------------------------------------------------
# 6) Fin
# ---------------------------------------------------------------------
print("\n✅ Exécution terminée.")
print(f"Consulte les fichiers dans '{REPORTS_DIR}/' pour les tableaux de résultats.")
