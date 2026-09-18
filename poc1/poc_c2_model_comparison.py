"""
PoC — HapSleep C2: Comparación LR / Random Forest / XGBoost
=============================================================

Complementa poc_c2_random_forest.py (el modelo elegido para C2 es RF, por
interpretabilidad — decisión ya cerrada, ver README). Este script existe
solo para documentar en el TI cómo se compara RF contra Regresión Logística
y XGBoost, usando el MISMO pipeline limpio (sin fuga de datos, ver
build_risk_label en poc_c2_random_forest.py) y la MISMA metodología robusta
(Repeated Stratified 5-Fold, 20 repeticiones) que ya se usa para reportar
las métricas oficiales de RF.

Por qué no reusar el F1=0.248 de la regresión logística que ya habían
corrido antes: ese número (igual que el F1=0.25 original de RF) salió de
una sola corrida de CV. Ya se demostró en este PoC que una sola partición
con n=71 tiene varianza enorme (el F1 de RF osciló entre 0.29 y 0.61 solo
por semilla) — comparar un LR de una sola corrida contra un RF promediado
en 20 repeticiones no sería una comparación justa. Aquí los tres modelos
se evalúan exactamente igual.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from xgboost import XGBClassifier

import poc_c2_random_forest as poc

RANDOM_STATE = poc.RANDOM_STATE
N_FOLDS = poc.N_FOLDS
N_REPEATS = poc.N_REPEATS


def make_lr(seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
    )


def make_xgb(seed, y):
    neg, pos = (y == 0).sum(), (y == 1).sum()
    return XGBClassifier(
        n_estimators=200,
        max_depth=3,             # n chica -> profundidad chica, mismo criterio que max_depth=5 del RF
        learning_rate=0.1,
        scale_pos_weight=neg / pos,  # equivalente a class_weight="balanced"
        eval_metric="logloss",
        random_state=seed,
        verbosity=0,
    )


def metrics(y_true, y_pred, y_proba):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def evaluate_model(model_factory, X, y, n_repeats=N_REPEATS):
    per_repeat = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE + r)
        model = model_factory(RANDOM_STATE, y) if model_factory is make_xgb else model_factory(RANDOM_STATE)
        proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
        per_repeat.append(metrics(y, pred, proba))
    names = per_repeat[0].keys()
    return {
        **{f"{m}_mean": float(np.mean([r[m] for r in per_repeat])) for m in names},
        **{f"{m}_std": float(np.std([r[m] for r in per_repeat])) for m in names},
    }


def main():
    raw_df = poc.load_data()
    person_df, feature_cols = poc.engineer_features(raw_df)
    y, label_source_raw = poc.build_risk_label(person_df)
    feature_cols = [c for c in feature_cols if not c.startswith(label_source_raw + "_")]
    X = person_df[feature_cols]

    print(f"\n[i] n personas = {len(person_df)} | features = {len(feature_cols)} "
          f"| prevalencia riesgo alto = {y.mean():.1%}")
    print(f"[i] Repeated Stratified {N_FOLDS}-Fold CV, {N_REPEATS} repeticiones, mismo pipeline sin fuga para los 3 modelos\n")

    models = {
        "logistic_regression": lambda seed: make_lr(seed),
        "random_forest": lambda seed: poc.make_rf(seed),
        "xgboost": make_xgb,
    }

    results = {}
    for name, factory in models.items():
        print(f"Corriendo {name}...")
        results[name] = evaluate_model(factory, X, y)

    print("\n" + "─" * 78)
    print(f"{'Modelo':22s} {'F1':>16s} {'ROC-AUC':>16s} {'PR-AUC':>16s}")
    print("─" * 78)
    for name, r in results.items():
        f1_s = f"{r['f1_mean']:.3f} ± {r['f1_std']:.3f}"
        roc_s = f"{r['roc_auc_mean']:.3f} ± {r['roc_auc_std']:.3f}"
        pr_s = f"{r['pr_auc_mean']:.3f} ± {r['pr_auc_std']:.3f}"
        print(f"{name:22s} {f1_s:>16s} {roc_s:>16s} {pr_s:>16s}")
    print("─" * 78)

    for name, r in results.items():
        print(f"\n{name} (detalle completo):")
        for metric in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
            print(f"   {metric:10s}: {r[f'{metric}_mean']:.3f} ± {r[f'{metric}_std']:.3f}")


if __name__ == "__main__":
    main()
