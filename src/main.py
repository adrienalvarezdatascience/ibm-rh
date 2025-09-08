import json
import numpy as np
import pandas as pd

from preprocessing import load_data
from modele_prediction_depart import (
    train_and_compare as entrainer_et_comparer_attrition,
    cost_optimal_threshold as seuil_cout_metier,
)
from modele_segmentation_employes import (
    build_matrix as construire_matrice,
    fit_kmeans as entrainer_kmeans,
    profile_clusters as profiler_clusters,
    score_range as evaluer_k,
)
from modele_indicateur_retention import (
    train_regression as entrainer_regression_stabilite,
    top_factors_shap_aggregated as facteurs_shap_top,
)

# Jeu de données
df = load_data("WA_Fn-UseC_-HR-Employee-Attrition.csv")
np.random.seed(42)

print(f"Jeu de données chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes.")


def colonnes_categorique_numerique(df_: pd.DataFrame):
    cat = [c for c in df_.columns if df_[c].dtype == "object"]
    num = [c for c in df_.columns if c not in cat]
    return cat, num


# 1) Prédiction des départs
if "Attrition" in df.columns:
    print("\n[Attrition] Entraînement et comparaison des modèles…")
    y_attr = df["Attrition"].values
    X_attr = df.drop(columns=["Attrition"])
    cat_cols, num_cols = colonnes_categorique_numerique(X_attr)

    resultats, meilleur_pipe = entrainer_et_comparer_attrition(
        X_attr, y_attr, cat_cols, num_cols
    )

    lignes = []
    for nom, res in resultats.items():
        lignes.append(
            {
                "Modele": nom,
                "PR_AUC": round(res.pr_auc, 3),
                "F1@0.5": round(res.f1_at_05, 3),
                "Meilleur_F1": round(res.best_f1, 3),
                "Seuil_F1": round(res.best_thr, 3),
                "Brier": round(res.brier, 3),
            }
        )
    df_scores_attr = pd.DataFrame(lignes).sort_values("PR_AUC", ascending=False)
    df_scores_attr.to_csv("attrition_metrics.csv", index=False)

    nom_top = df_scores_attr.iloc[0]["Modele"]
    proba_oof = resultats[nom_top].proba_oof
    seuil_opt, cout_min = seuil_cout_metier(proba_oof, y_attr, cost_fn=10.0, cost_fp=1.0)
    with open("attrition_threshold.json", "w") as f:
        json.dump(
            {
                "meilleur_modele": nom_top,
                "seuil_optimal": float(seuil_opt),
                "cout_min": float(cout_min),
            },
            f,
            indent=2,
        )

    print("→ Fichiers générés : attrition_metrics.csv, attrition_threshold.json")


# 2) Segmentation employés
print("\n[Segmentation] KMeans et profils de clusters…")
df_clust = df.drop(columns=["Attrition"]) if "Attrition" in df.columns else df.copy()
cat_c, num_c = colonnes_categorique_numerique(df_clust)

X_mat, prepro_clust = construire_matrice(df_clust, cat_c, num_c)

df_k = evaluer_k(X_mat, k_min=2, k_max=8)
df_k.to_csv("clustering_k_scores.csv", index=False)

km2 = entrainer_kmeans(X_mat, k=2)
profils_k2 = profiler_clusters(df_clust, km2.labels_, cat_c, num_c)
profils_k2.to_csv("clustering_k2_profiles.csv", index=False)

km5 = entrainer_kmeans(X_mat, k=5)
profils_k5 = profiler_clusters(df_clust, km5.labels_, cat_c, num_c)
profils_k5.to_csv("clustering_k5_profiles.csv", index=False)

print("→ Fichiers générés : clustering_k_scores.csv, clustering_k2_profiles.csv, clustering_k5_profiles.csv")


# 3) Indicateur de rétention
if "YearsAtCompany" in df.columns:
    print("\n[Stabilité] Régression YearsAtCompany + SHAP Top-10…")
    y_yac = df["YearsAtCompany"].values
    X_yac = df.drop(columns=["YearsAtCompany"])
    cat_s, num_s = colonnes_categorique_numerique(X_yac)

    df_scores_stab, pipe_stab, yhat_yac = entrainer_regression_stabilite(
        X_yac, y_yac, cat_s, num_s
    )
    df_scores_stab.to_csv("stability_scores.csv", index=False)

    top10_stab = facteurs_shap_top(X_yac, y_yac, cat_s, num_s, task="reg", topn=10)
    top10_stab.to_csv("top10_stability.csv", index=False)

    if "Attrition" in df.columns:
        X_attr2 = df.drop(columns=["Attrition"])
        y_attr2 = df["Attrition"].values
        cat_a, num_a = colonnes_categorique_numerique(X_attr2)

        top10_attr = facteurs_shap_top(X_attr2, y_attr2, cat_a, num_a, task="clf", topn=10)
        top10_attr.to_csv("top10_attrition.csv", index=False)

        communs = sorted(
            set(top10_stab["base_feat"]).intersection(set(top10_attr["base_feat"]))
        )
        with open("top10_common_stability_attrition.json", "w") as f:
            json.dump({"variables_communes": list(communs)}, f, indent=2)

    print("→ Fichiers générés : stability_scores.csv, top10_stability.csv (+ top10_attrition.csv si dispo)")


# Fin
print("\nTerminé.")
