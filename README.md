# IBM RH

J’ai monté ce projet pour préparer un entretien avec la Marine nationale (section ITN).
L’objectif : montrer, sur un cas concret, comment j’utilise la data pour des sujets RH.

<img width="550" height="284" alt="image" src="https://github.com/user-attachments/assets/32811649-0eb6-4a86-85c2-f21029d84954" />

---

## Objectifs

- Anticiper les départs.  
- Mieux comprendre les profils.  
- Identifier ce qui favorise la rétention.

**Données** : dataset public *IBM HR Analytics Employee Attrition & Performance* (Kaggle).

---

## Contenu du projet

### 1) Indicateur rétention (classification)
- **But** : repérer tôt les personnels à risque pour déclencher un soutien ciblé (moral, famille, affectation).  
- **Méthode** : régression logistique (comparée à arbre, Random Forest, KNN), validation croisée, métriques adaptées au déséquilibre (PR-AUC, F1).  
- **Pratique** : choix d’un **seuil métier** (mode *préventif* vs *très sûr*).

### 2) Segmentation employés (clustering)
- **But** : regrouper les profils pour adapter les actions RH.  
- **Méthodes** : K-Means, visualisation PCA.

### 3) Stabilité (régression)
- **But** : comprendre ce qui retient les personnes (ancienneté / *YearsAtCompany*).  
- **Méthode** : modèles Linéaire, Arbre, Random Forest.
