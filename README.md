# IBM HR Analytics

## 🎯 Objectif du projet
J’ai réalisé ce projet pour préparer un entretien à la **Direction du Personnel de la Marine nationale (DRH-M)**, plus précisément pour la section **Innovation et Transformation Numérique (ITN)**.  
L’idée est de montrer, à travers un cas pratique, ma capacité à utiliser la data science pour répondre à des enjeux RH concrets :  
- anticiper les départs,  
- mieux comprendre les profils du personnel,  
- analyser la masse salariale et l’équité,  
- identifier les leviers de stabilité et de fidélisation.  

## 📊 Jeu de données
Le jeu de données utilisé est **IBM HR Analytics Employee Attrition & Performance**, un dataset disponible sur Kaggle.  
Il a été adapté ici comme **maquette** pour des problématiques proches de celles que pourrait rencontrer la Marine nationale.

## 🧩 Cas d’usage développés
1. **Attrition (classification)**  
   - But : détecter les personnels à risque de départ afin de déclencher un soutien ciblé (moral, famille, affectation).  
   - Modèles : Régression logistique, Arbre de décision, Random Forest, KNN.  
   - Optimisation : recherche d’hyperparamètres + métriques adaptées (PR-AUC, F1).  
   - Explicabilité : SHAP + scénarios “what-if”.  

2. **Segmentation (clustering)**  
   - But : regrouper les profils en segments utiles pour le pilotage RH (ex : jeunes en surcharge, cadres en milieu de carrière).  
   - Méthodes : K-Means + PCA pour la visualisation.  
   - Résultat : clusters transformés en segments lisibles pour les décideurs RH.  

3. **Stabilité (YearsAtCompany)**  
   - But : comprendre ce qui retient le personnel (ancienneté dans l’organisation).  
   - Approche : prédiction en cross-validation + SHAP.  
   - Résultat : les variables d’expérience (JobLevel, YearsWithCurrManager, TotalWorkingYears) sont déterminantes.  
   - Comparaison avec l’attrition → double vision : “ce qui retient” vs “ce qui pousse à partir”.  

## 🛠️ Environnement technique
- **Langages** : Python 3  
- **Bibliothèques principales** : scikit-learn, pandas, matplotlib, shap  
- **Organisation du code** :  
  - `src/` → modules Python (prétraitement, modèles, main.py)  
  - `notebooks/` → notebooks de résultats (visualisations + explications)  
  - `report/` → bilan du projet 

## 🚀 Comment lancer le projet
1. Cloner le dépôt  
   ```bash
   git clone https://github.com/adrienalvarezdatascience/ibm-hr.git
   cd ibm-hr

2. Installer les dépendances
  ```bash
  pip install -r requirements.txt```

4. Lancer le script principal
   python src/main.py

6. Consulter les résultats générés dans le dossier reports/.
