# Prédire la réhospitalisation à 30 jours : régression logistique contre gradient boosting

Ce projet compare deux modèles de prédiction du risque de réhospitalisation dans les 30 jours suivant une sortie d'hôpital, avec une démarche d'évaluation rigoureuse : jeu de test mis de côté avant toute modélisation, validation croisée, intervalle de confiance de l'AUC, calibration et importance des variables.

Les données sont **simulées** (6 000 séjours), à partir d'un vrai modèle de risque connu, qui inclut des effets non linéaires et une interaction. Aucune donnée de patient n'est utilisée.

Auteur : Armel Mahutin KANHONOU · [Portfolio](https://armel-mahutin.github.io) · ORCID [0009-0005-1764-1327](https://orcid.org/0009-0005-1764-1327)

## Variables

Âge, sexe, durée de séjour, nombre d'hospitalisations dans l'année, score de Charlson, diabète, insuffisance cardiaque, personne vivant seule, précarité, mode d'entrée (urgences ou programmé). Taux de réhospitalisation simulé : 27,5 %. Deux variables ont 5 % de valeurs manquantes.

## Démarche

1. **Jeu de test (25 %)** mis de côté dès le départ, stratifié sur l'issue.
2. **Prétraitement dans un pipeline** (imputation par la médiane, standardisation, encodage) : il est appris sur les seules données d'apprentissage de chaque pli, ce qui évite toute fuite de données.
3. **Validation croisée stratifiée à 5 plis** sur le jeu d'apprentissage.
4. **Évaluation finale sur le jeu de test** : AUC avec intervalle de confiance à 95 % (1 000 rééchantillonnages bootstrap), score de Brier, courbe de calibration.
5. **Comparaison directe des deux modèles** : différence d'AUC avec un bootstrap apparié (2 000 rééchantillonnages, les mêmes patients tirés pour les deux modèles), ce qui tient compte de la corrélation entre leurs prédictions.
6. **Importance des variables par permutation** sur le jeu de test.

## Résultats

| Modèle | AUC en validation croisée | AUC sur le jeu de test (IC 95 %) | Score de Brier (test) |
|---|---|---|---|
| Régression logistique | 0,702 (± 0,012) | **0,709** (0,681 – 0,742) | **0,176** |
| Gradient boosting | 0,690 (± 0,017) | 0,697 (0,668 – 0,729) | 0,179 |
| Référence : taux moyen | – | 0,500 | 0,200 |

**Différence d'AUC (logistique moins boosting) :** +0,012 (IC 95 % : −0,003 à +0,027). L'intervalle contient zéro : l'écart entre les deux modèles peut relever du hasard, et le boosting ne fait mieux que dans 6 % des rééchantillonnages.

**Lecture :** le gradient boosting ne fait pas mieux que la régression logistique, alors qu'il est plus complexe et moins interprétable. Avec quelques variables cliniques bien choisies, un modèle simple et bien calibré suffit souvent. Les deux modèles restent modérément discriminants (AUC autour de 0,70), ce qui est habituel pour la réhospitalisation.

![Discrimination et calibration](figures/1%20discrimination%20et%20calibration.png)

![Importance des variables](figures/2%20importance%20des%20variables.png)

## Limites

- Données simulées : le but est de montrer la démarche d'évaluation, pas de produire un outil clinique.
- Pas de validation externe (autre établissement, autre période), indispensable avant tout usage réel.
- Les seuils de décision (qui contacter après la sortie ?) dépendent des ressources disponibles et seraient à fixer avec les équipes, par exemple par une analyse de la courbe de décision.

## Reproduire l'analyse

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python src/simuler.py
python src/modeliser.py
```

## Organisation

```
src/simuler.py      simulation de la cohorte et du vrai modèle de risque
src/modeliser.py    pipelines, validation croisée, évaluation, figures
donnees/            cohorte simulée
resultats/          performances (JSON)
figures/            figures
```
