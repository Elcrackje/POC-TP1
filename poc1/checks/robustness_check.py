"""
Chequeo de robustez (archivo histórico — NO forma parte del pipeline oficial
de C2). Documenta cómo se encontraron dos cosas:

  1. Que un solo Stratified 5-Fold (una sola semilla) tiene varianza enorme
     con n=71 — por eso poc_c2_random_forest.py pasó a Repeated Stratified
     K-Fold (20 repeticiones).
  2. Que agregar STAI (ansiedad) y PANAS (afecto) como features psicológicas
     parecía mejorar mucho el F1 con una sola semilla, pero esa "mejora" no
     se sostuvo bajo repeated CV — por eso NO están en el pipeline final
     (ver README.md, sección de limitaciones, punto 6).

Este script es autocontenido (no importa load_survey_features de
poc_c2_random_forest.py porque esa función ya no existe ahí — se quitó del
pipeline oficial precisamente por lo que este chequeo encontró). Se guarda
para que el hallazgo sea reproducible, no como código vivo.
"""
import sys
sys.path.insert(0, r"C:\develop\tesis\poc1")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

import poc_c2_random_forest as poc

STAI_PATH = poc.Path("rais_anonymized/rais_anonymized/scored_surveys/stai.csv")
PANAS_PATH = poc.Path("rais_anonymized/rais_anonymized/scored_surveys/panas.csv")


def load_survey_features():
    frames = []
    for path, cols in [(STAI_PATH, ["stai_stress"]), (PANAS_PATH, ["positive_affect_score", "negative_affect_score"])]:
        sdf = pd.read_csv(path)
        for col in cols:
            sdf[col] = pd.to_numeric(sdf[col], errors="coerce")
        agg = sdf.groupby("user_id")[cols].agg(["mean", "std"])
        agg.columns = [f"psych_{col}_{stat}" for col, stat in agg.columns]
        frames.append(agg)
    survey_df = frames[0].join(frames[1:], how="outer")
    survey_df.index.name = poc.COL_ID
    return survey_df.reset_index()


raw_df = poc.load_data()
person_df, feature_cols = poc.engineer_features(raw_df)
survey_df = load_survey_features()
person_df = person_df.merge(survey_df, on=poc.COL_ID, how="left")
psych_cols = [c for c in survey_df.columns if c != poc.COL_ID]
person_df[psych_cols] = person_df[psych_cols].fillna(person_df[psych_cols].median())
feature_cols = feature_cols + psych_cols

y, label_source_raw = poc.build_risk_label(person_df)
feature_cols = [c for c in feature_cols if not c.startswith(label_source_raw + "_")]

physio_cols = [c for c in feature_cols if not c.startswith("psych_")]


def evaluate(cols, n_repeats=20, n_splits=5, label=""):
    X = person_df[cols]
    f1s, rocs, prs = [], [], []
    for seed in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1000 + seed)
        rf = RandomForestClassifier(n_estimators=300, max_depth=5, class_weight="balanced", random_state=42)
        proba = cross_val_predict(rf, X, y, cv=skf, method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
        f1s.append(f1_score(y, pred, zero_division=0))
        rocs.append(roc_auc_score(y, proba))
        prs.append(average_precision_score(y, proba))
    print(f"\n=== {label} (n_features={len(cols)}) — {n_repeats} repeticiones de Stratified {n_splits}-Fold ===")
    print(f"F1:      mean={np.mean(f1s):.3f}  std={np.std(f1s):.3f}  min={np.min(f1s):.3f}  max={np.max(f1s):.3f}")
    print(f"ROC-AUC: mean={np.mean(rocs):.3f}  std={np.std(rocs):.3f}  min={np.min(rocs):.3f}  max={np.max(rocs):.3f}")
    print(f"PR-AUC:  mean={np.mean(prs):.3f}  std={np.std(prs):.3f}  min={np.min(prs):.3f}  max={np.max(prs):.3f}")


if __name__ == "__main__":
    evaluate(physio_cols, label="SOLO fisiologicas/conductuales (baseline)")
    evaluate(feature_cols, label="CON psicologicas (STAI/PANAS) agregadas")
