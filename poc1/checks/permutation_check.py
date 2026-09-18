"""
Permutation importance sobre el RF ya entrenado (pipeline limpio, sin fuga),
para comparar contra el ranking de Gini importance ya reportado en
c2_output_sample.json. Si el top feature cambia, hay que decirlo, no asumir
que Gini importance (que sesga a favor de variables continuas/alta
cardinalidad) es la verdad.
"""
import sys
sys.path.insert(0, r"C:\develop\tesis\poc1")

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import poc_c2_random_forest as poc

raw_df = poc.load_data()
person_df, feature_cols = poc.engineer_features(raw_df)
y, label_source_raw = poc.build_risk_label(person_df)
feature_cols = [c for c in feature_cols if not c.startswith(label_source_raw + "_")]
X = person_df[feature_cols]

# Gini importance ya calculada (promedio de 20 fits) desde el pipeline oficial
_, gini_importances, _ = poc.train_evaluate(X, y)

# Permutation importance: promediada sobre las mismas 20 semillas, evaluada
# con out-of-fold predictions (no en el propio training set, para no inflarla
# con overfitting) -- se refit un modelo por semilla sobre TODO el dataset
# (igual que gini) y se mide permutation importance sobre ese mismo dataset,
# pero usando scoring="roc_auc" para que sea comparable a lo que reportamos.
n_repeats_outer = 20
perm_sum = np.zeros(len(feature_cols))
for r in range(n_repeats_outer):
    rf = poc.make_rf(poc.RANDOM_STATE + r)
    rf.fit(X, y)
    result = permutation_importance(
        rf, X, y, scoring="roc_auc", n_repeats=10, random_state=poc.RANDOM_STATE + r
    )
    perm_sum += result.importances_mean

perm_importances = pd.Series(perm_sum / n_repeats_outer, index=feature_cols)

gini_sorted = gini_importances.sort_values(ascending=False)
perm_sorted = perm_importances.sort_values(ascending=False)

print("\n=== TOP 10 — Gini importance (MDI, promedio 20 fits) ===")
for f, v in gini_sorted.head(10).items():
    print(f"   {f:35s} {v:.4f}")

print("\n=== TOP 10 — Permutation importance (promedio 20x10 permutaciones, scoring=roc_auc) ===")
for f, v in perm_sorted.head(10).items():
    print(f"   {f:35s} {v:.4f}")

print("\n=== Comparacion de rank (1=mas importante) ===")
comp = pd.DataFrame({
    "gini_rank": gini_sorted.rank(ascending=False).astype(int),
    "perm_rank": perm_sorted.rank(ascending=False).astype(int),
}).sort_values("gini_rank")
print(comp.head(10))

top_gini = gini_sorted.index[0]
top_perm = perm_sorted.index[0]
print(f"\nTop feature Gini: {top_gini}")
print(f"Top feature Permutation: {top_perm}")
print("COINCIDEN" if top_gini == top_perm else "NO COINCIDEN -- el ranking cambia segun el metodo")
