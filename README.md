# IBM RH

J'ai monté ce projet pour préparer un entretien avec la Marine Nationale (section ITN). L’objectif est de montrer sur un cas concret, comment j’utilise la data pour des sujets RH.

<img width="550" height="284" alt="image" src="https://github.com/user-attachments/assets/32811649-0eb6-4a86-85c2-f21029d84954" />


Ce que je cherche à faire :
- Anticiper les départs.
- Mieux comprendre les profils.
- Identifier ce qui fait la rétention des employés.

Pour ce faire j’utilise le dataset public “IBM HR Analytics Employee Attrition & Performance”.

Vous pourrez retrouver :

Indicateur rétention (classification)
Objectif : repérer tôt les personnels à risque pour déclencher un soutien ciblé (moral, famille, affectation).
Méthode : régression logistique surtout (comparée à arbre, Random Forest, KNN), validation croisée, métriques adaptées aux classes déséquilibrées (PR-AUC, F1).
Pratique : choix d’un seuil “métier” (mode préventif vs très sûr).

Segmentation employés (clustering)
Objectif : regrouper les profils pour adapter les actions RH.
Méthodes : K-Means, visualisation PCA.

Stabilité (regression)
Objectif : comprendre ce qui retient les personnes.
Méthode : Linéaire, Arbre, Random Forest.
