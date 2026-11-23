"""Compare une régression logistique et un modèle de gradient boosting pour
prédire la réhospitalisation à 30 jours : validation croisée, discrimination,
calibration et importance des variables, sans fuite de données."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

d = pd.read_csv("donnees/cohorte simulee.csv")
y = d.pop("rehospit_30j")
num = ["age", "duree_sejour", "nb_hospit_an", "charlson"]
bin_ = ["diabete", "insuffisance_cardiaque", "vit_seul", "precarite"]
cat = ["sexe", "mode_entree"]

# Jeu de test mis de côté AVANT toute modélisation
X_app, X_test, y_app, y_test = train_test_split(d, y, test_size=0.25, stratify=y,
                                                random_state=2026)

prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("std", StandardScaler())]), num),
    ("bin", "passthrough", bin_),
    ("cat", OneHotEncoder(drop="first"), cat)])

modeles = {
    "Régression logistique": Pipeline([("prep", prep),
                                       ("m", LogisticRegression(max_iter=1000))]),
    "Gradient boosting": Pipeline([("prep", prep),
                                   ("m", HistGradientBoostingClassifier(
                                       learning_rate=0.05, max_iter=300,
                                       max_leaf_nodes=15, l2_regularization=1.0,
                                       early_stopping=True, random_state=2026))]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
resultats = {}
for nom, pipe in modeles.items():
    s = cross_validate(pipe, X_app, y_app, cv=cv,
                       scoring={"auc": "roc_auc", "brier": "neg_brier_score"})
    pipe.fit(X_app, y_app)
    p = pipe.predict_proba(X_test)[:, 1]
    # IC 95 % de l'AUC de test par bootstrap
    rng = np.random.default_rng(1)
    boots = []
    for _ in range(1000):
        i = rng.integers(0, len(y_test), len(y_test))
        if y_test.iloc[i].nunique() == 2:
            boots.append(roc_auc_score(y_test.iloc[i], p[i]))
    resultats[nom] = dict(
        auc_cv=round(s["test_auc"].mean(), 3), auc_cv_et=round(s["test_auc"].std(), 3),
        brier_cv=round(-s["test_brier"].mean(), 4),
        auc_test=round(roc_auc_score(y_test, p), 3),
        auc_test_ic=[round(float(np.percentile(boots, 2.5)), 3),
                     round(float(np.percentile(boots, 97.5)), 3)],
        brier_test=round(brier_score_loss(y_test, p), 4))
    resultats[nom]["_p"] = p

# Différence d'AUC entre les deux modèles : bootstrap APPARIÉ (mêmes patients
# tirés pour les deux modèles), pour savoir si l'écart dépasse le hasard
p_lr, p_gb = resultats["Régression logistique"]["_p"], resultats["Gradient boosting"]["_p"]
rng = np.random.default_rng(7)
diffs = []
for _ in range(2000):
    i = rng.integers(0, len(y_test), len(y_test))
    if y_test.iloc[i].nunique() == 2:
        diffs.append(roc_auc_score(y_test.iloc[i], p_lr[i]) - roc_auc_score(y_test.iloc[i], p_gb[i]))
diffs = np.array(diffs)
diff_obs = roc_auc_score(y_test, p_lr) - roc_auc_score(y_test, p_gb)
difference = dict(
    lr_moins_gb=round(float(diff_obs), 3),
    ic95=[round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3)],
    part_bootstrap_gb_meilleur=round(float((diffs < 0).mean()), 3),
    n_bootstrap=int(len(diffs)))

base = y_test.mean()
brier_ref = round(brier_score_loss(y_test, np.full(len(y_test), y_app.mean())), 4)
sortie = {k: {kk: vv for kk, vv in v.items() if kk != "_p"} for k, v in resultats.items()}
sortie["Référence (taux moyen)"] = {"brier_test": brier_ref}
sortie["Différence d'AUC (logistique moins boosting), bootstrap apparié"] = difference
json.dump(sortie, open("resultats/performances.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(json.dumps(sortie, ensure_ascii=False, indent=2))

coul = {"Régression logistique": "#2b5d63", "Gradient boosting": "#c0583e"}
# Figure 1 : courbes ROC et calibration
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for nom, r in resultats.items():
    fpr, tpr, _ = roc_curve(y_test, r["_p"])
    ax[0].plot(fpr, tpr, color=coul[nom], lw=2, label=f"{nom} (AUC {r['auc_test']:.2f})")
    frac, moy = calibration_curve(y_test, r["_p"], n_bins=10, strategy="quantile")
    ax[1].plot(moy, frac, "o-", color=coul[nom], lw=2, label=nom)
ax[0].plot([0, 1], [0, 1], "--", color="grey"); ax[0].set(xlabel="1 - spécificité",
          ylabel="Sensibilité", title="Discrimination (jeu de test)")
lim = max(ax[1].get_xlim()[1], ax[1].get_ylim()[1])
ax[1].plot([0, lim], [0, lim], "--", color="grey")
ax[1].set(xlabel="Risque prédit", ylabel="Fréquence observée",
          title="Calibration (déciles de risque)")
for a in ax: a.legend(frameon=False, fontsize=9); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figures/1 discrimination et calibration.png", dpi=170)

# Figure 2 : importance des variables par permutation (modèle logistique)
pi = permutation_importance(modeles["Régression logistique"], X_test, y_test,
                            scoring="roc_auc", n_repeats=30, random_state=2026)
imp = pd.Series(pi.importances_mean, index=X_test.columns).sort_values()
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.barh(imp.index, imp.values, color="#3f7f86")
ax.set(xlabel="Perte d'AUC quand la variable est permutée",
       title="Importance des variables (régression logistique)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figures/2 importance des variables.png", dpi=170)
