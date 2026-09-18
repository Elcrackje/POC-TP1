"""
PoC 2 — ML1: Random Forest de riesgo de calidad de sueño sobre datos SINTÉTICOS
=================================================================================

Mismo pipeline y misma disciplina metodológica que poc1 (Repeated Stratified
5-Fold, PR-AUC/ROC-AUC/F1 contra baseline trivial, chequeo de fuga de datos
en la etiqueta), pero corriendo sobre datos 100% sintéticos generados por
generate_synthetic_wearable_data.py — no sobre LifeSnaps.

Por qué correr esto además de poc1 (que ya usa datos reales): acá SÍ existe
un "riesgo verdadero" (latent_risk) por persona, guardado en
synthetic_ground_truth.csv y NUNCA usado para entrenar el modelo. Eso
permite una validación que con datos reales es imposible: comparar qué tan
bien el risk_probability que predice el modelo se parece al riesgo real
que se inyectó al generar los datos, no solo contra la etiqueta proxy
(que en poc1 es sleep_points_percentage, y aquí también lo es).

IMPORTANTE — qué SÍ y qué NO demuestra este PoC:
  - SÍ demuestra que el pipeline (agregación a nivel de persona, feature
    engineering, Random Forest, evaluación) recupera una señal de riesgo
    conocida cuando existe.
  - NO reemplaza los resultados de poc1: el desempeño en datos sintéticos
    siempre va a verse mejor que en datos reales, porque el generador
    inyecta una relación limpia (con ruido controlado) entre features y
    riesgo -- la vida real tiene missing data, medición imperfecta, y
    factores que el generador no modela. No cites estos números como
    "el modelo funciona con X% de desempeño" en el TI -- son de
    validación de pipeline, no de desempeño esperado en producción.
"""
import sys
import warnings
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score, f1_score,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_FOLDS = 5
N_REPEATS = 20               # misma disciplina que poc1: un solo K-Fold no alcanza con n chica
RISK_LABEL_QUANTILE = 0.25

DATA_PATH = Path("synthetic_wearable_data.csv")
GROUND_TRUTH_PATH = Path("synthetic_ground_truth.csv")

COL_ID = "id"
COL_DATE = "date"

FEATURE_COLUMNS_PHYSIO = [
    "rmssd", "nremhr", "resting_hr", "spo2", "full_sleep_breathing_rate",
    "minutesToFallAsleep", "minutesAsleep", "minutesAwake",
    "sleep_efficiency", "sleep_points_percentage",
]
FEATURE_COLUMNS_BEHAVIORAL = ["steps", "sedentary_minutes"]
ALL_RAW_COLUMNS = FEATURE_COLUMNS_PHYSIO + FEATURE_COLUMNS_BEHAVIORAL


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No encuentro {DATA_PATH}. Corre generate_synthetic_wearable_data.py primero.")
    return pd.read_csv(DATA_PATH)


def engineer_features(df: pd.DataFrame):
    present_cols = [c for c in ALL_RAW_COLUMNS if c in df.columns]
    agg = df.groupby(COL_ID)[present_cols].agg(["mean", "std"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()

    if "minutesAsleep_std" in agg.columns:
        agg = agg.rename(columns={"minutesAsleep_std": "sleep_duration_inconsistency"})

    feature_cols = [c for c in agg.columns if c != COL_ID]
    agg[feature_cols] = agg[feature_cols].fillna(agg[feature_cols].median())
    return agg, feature_cols


def build_risk_label(person_df: pd.DataFrame):
    """Mismo chequeo de fuga que poc1: la columna usada acá se excluye de X en main()."""
    quality_col = "sleep_points_percentage_mean"
    threshold = person_df[quality_col].quantile(RISK_LABEL_QUANTILE)
    y = (person_df[quality_col] <= threshold).astype(int)
    raw_source_name = quality_col.rsplit("_", 1)[0]
    return y, raw_source_name


def make_rf(seed):
    return RandomForestClassifier(
        n_estimators=300, max_depth=5, class_weight="balanced", random_state=seed,
    )


def train_evaluate(X: pd.DataFrame, y: pd.Series, n_repeats: int = N_REPEATS):
    baseline_pred = DummyClassifier(strategy="most_frequent").fit(X, y).predict(X)

    def metrics(y_true, y_pred, y_proba=None):
        out = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
        if y_proba is not None:
            out["roc_auc"] = roc_auc_score(y_true, y_proba)
            out["pr_auc"] = average_precision_score(y_true, y_proba)
        return out

    per_repeat = []
    proba_sum = np.zeros(len(y))
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE + r)
        proba = cross_val_predict(make_rf(RANDOM_STATE), X, y, cv=skf, method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
        per_repeat.append(metrics(y, pred, proba))
        proba_sum += proba

    metric_names = per_repeat[0].keys()
    rf_stats = {f"{m}_mean": float(np.mean([r[m] for r in per_repeat])) for m in metric_names}
    rf_stats.update({f"{m}_std": float(np.std([r[m] for r in per_repeat])) for m in metric_names})

    results = {
        "random_forest": rf_stats,
        "baseline_trivial": metrics(y, baseline_pred),
        "prevalencia_clase_riesgo": float(y.mean()),
    }

    mean_risk_proba = proba_sum / n_repeats

    importances_sum = np.zeros(X.shape[1])
    for r in range(n_repeats):
        rf_full = make_rf(RANDOM_STATE + r)
        rf_full.fit(X, y)
        importances_sum += rf_full.feature_importances_
    mean_importances = pd.Series(importances_sum / n_repeats, index=X.columns)

    return results, mean_importances, mean_risk_proba


def validate_against_ground_truth(person_df, risk_proba, y_proxy):
    """
    Esto es lo que NO se puede hacer con datos reales: comparar la predicción
    del modelo contra el riesgo verdadero que se inyectó al generar los
    datos (latent_risk), nunca visto por el modelo.
    """
    gt = pd.read_csv(GROUND_TRUTH_PATH)
    merged = person_df[[COL_ID]].merge(gt, on=COL_ID, how="left")
    latent_risk = merged["latent_risk"].values

    corr, pval = spearmanr(risk_proba, latent_risk)

    # Etiqueta "verdadera" alternativa: top 25% de latent_risk (en vez del proxy sleep_points_percentage)
    true_high_risk = (latent_risk >= np.quantile(latent_risk, 1 - RISK_LABEL_QUANTILE)).astype(int)
    roc_auc_vs_truth = roc_auc_score(true_high_risk, risk_proba)

    # Qué tan de acuerdo están la etiqueta proxy (basada en sleep_points_percentage, con ruido de medición)
    # y la etiqueta "verdadera" (basada en latent_risk directo) -- si difieren mucho, el proxy no es
    # tan buen sustituto de la verdad como uno esperaría, ni siquiera en un mundo sintético limpio.
    agreement = float((y_proxy.values == true_high_risk).mean())

    print("\n" + "─" * 60)
    print("VALIDACIÓN CONTRA GROUND TRUTH (solo posible con datos sintéticos)")
    print("─" * 60)
    print(f"Correlación de Spearman (risk_probability vs. latent_risk real): {corr:.3f} (p={pval:.4f})")
    print(f"ROC-AUC del modelo contra la etiqueta 'verdadera' (top 25% latent_risk): {roc_auc_vs_truth:.3f}")
    print(f"Acuerdo entre etiqueta proxy (sleep_points_percentage) y etiqueta verdadera (latent_risk): {agreement:.1%}")


def main():
    raw_df = load_data()
    person_df, feature_cols = engineer_features(raw_df)
    y, label_source_raw = build_risk_label(person_df)

    feature_cols = [c for c in feature_cols if not c.startswith(label_source_raw + "_")]
    X = person_df[feature_cols]

    print(f"\n[i] n personas = {len(person_df)} | features = {len(feature_cols)}")
    print(f"[i] Prevalencia de 'riesgo alto' (proxy): {y.mean():.1%}\n")

    results, importances, risk_proba = train_evaluate(X, y)

    print("─" * 60)
    print(f"RESULTADOS (Repeated Stratified {N_FOLDS}-Fold CV, {N_REPEATS} repeticiones) — DATOS SINTÉTICOS")
    print("─" * 60)
    print(f"\nrandom_forest (mean ± std sobre {N_REPEATS} repeticiones):")
    for metric in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
        mean_v = results["random_forest"][f"{metric}_mean"]
        std_v = results["random_forest"][f"{metric}_std"]
        print(f"   {metric:10s}: {mean_v:.3f} ± {std_v:.3f}")
    print("\nbaseline_trivial:")
    for metric, value in results["baseline_trivial"].items():
        print(f"   {metric:10s}: {value:.3f}")

    print("\nTop 10 feature importance (Gini, promedio 20 fits):")
    for f, v in importances.sort_values(ascending=False).head(10).items():
        print(f"   {f:35s} {v:.4f}")

    if GROUND_TRUTH_PATH.exists():
        validate_against_ground_truth(person_df, risk_proba, y)


if __name__ == "__main__":
    main()
