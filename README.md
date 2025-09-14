# IBM RH

J’ai monté ce projet en un week-end pour préparer un entretien avec la Marine nationale (section ITN).
L’objectif : montrer sur un cas concret comment j’utilise la data pour des sujets RH (énormément d'améliorations restent possible sur chacun des modèles, si besoin je peux développer ce point sans problème).

<img width="550" height="284" alt="image" src="https://github.com/user-attachments/assets/32811649-0eb6-4a86-85c2-f21029d84954" />

---

## Objectifs

- Anticiper les départs.  
- Mieux comprendre les profils.  
- Identifier ce qui favorise la rétention.

**Données** : dataset public *IBM HR Analytics Employee Attrition & Performance* (Kaggle).

---

## Contenu du projet

### 1) Prediction depart (classification)
- **But** : repérer tôt les personnels à risque pour déclencher un soutien ciblé (moral, famille, affectation).  
- **Méthode** : régression logistique (comparée à arbre, Random Forest, KNN), validation croisée, métriques adaptées au déséquilibre (PR-AUC, F1).  
- **Pratique** : choix d’un **seuil métier** (mode *préventif* vs *très sûr*).

### 2) Segmentation employés (clustering)
- **But** : regrouper les profils pour adapter les actions RH.  
- **Méthodes** : K-Means, visualisation PCA.

### 3) Indicateur rétention (regression)
- **But** : comprendre ce qui retient les personnes (ancienneté / *YearsAtCompany*).  
- **Méthode** : modèles Linéaire, Arbre, Random Forest.

---

## Lancer le projet

1. Cloner le dépôt
   ```bash
   git clone https://github.com/adrienalvarezdatascience/ibm-rh.git
   cd ibm-rh
   ```

2. Installer les dépendances
   ```bash
   pip install -r requirements.txt
   ```

4. Lancer le script principal
   ```bash
   python src/main.py
   ```
   
6. Consulter les résultats générés dans le dossier reports/.
