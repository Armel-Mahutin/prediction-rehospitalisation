"""Simule une cohorte de sorties d'hospitalisation avec un risque de
réhospitalisation à 30 jours connu, dépendant de variables cliniques et sociales."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
n = 6000

age = rng.normal(64, 15, n).clip(18, 98).round()
sexe = rng.choice(["F", "H"], n)
duree_sejour = rng.gamma(2.0, 3.0, n).round().clip(1, 60)
nb_hospit_an = rng.poisson(0.8, n)
charlson = rng.poisson(2.0, n).clip(0, 12)
diabete = rng.binomial(1, 0.22, n)
insuffisance_cardiaque = rng.binomial(1, 0.12 + 0.002 * (age - 64).clip(0), n)
vit_seul = rng.binomial(1, 0.30, n)
precarite = rng.binomial(1, 0.18, n)
mode_entree = rng.choice(["urgences", "programme"], n, p=[0.62, 0.38])

# Vrai modèle : effets non linéaires (âge, durée) et une interaction
logit = (-3.1
         + 0.025 * (age - 64) + 0.0008 * (age - 64) ** 2
         + 0.35 * np.log1p(duree_sejour)
         + 0.45 * nb_hospit_an
         + 0.18 * charlson
         + 0.30 * diabete + 0.55 * insuffisance_cardiaque
         + 0.25 * vit_seul + 0.40 * precarite
         + 0.35 * vit_seul * (age > 75)
         + 0.30 * (mode_entree == "urgences"))
p = 1 / (1 + np.exp(-logit))
rehospit_30j = rng.binomial(1, p)

d = pd.DataFrame(dict(age=age, sexe=sexe, duree_sejour=duree_sejour,
                      nb_hospit_an=nb_hospit_an, charlson=charlson, diabete=diabete,
                      insuffisance_cardiaque=insuffisance_cardiaque, vit_seul=vit_seul,
                      precarite=precarite, mode_entree=mode_entree,
                      rehospit_30j=rehospit_30j))
# 5 % de valeurs manquantes sur deux variables, comme dans des données réelles
for col in ["duree_sejour", "charlson"]:
    d.loc[rng.random(n) < 0.05, col] = np.nan
d.to_csv("cohorte simulee.csv", index=False)
print(f"{n} séjours, taux de réhospitalisation à 30 jours : {d.rehospit_30j.mean():.1%}")
