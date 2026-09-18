"""
PoC — HapSleep C2: Motor de Machine Learning (Random Forest)
==============================================================

Qué hace este script:
  1. Carga datos de wearable a granularidad diaria (LifeSnaps, Fitbit Sense, n=71 -
     el mismo dataset ya usado en el análisis de autocorrelación/estabilidad r=0.727).
  2. Agrega a NIVEL DE PERSONA (no de noche puntual) — esto replica a propósito
     el hallazgo ya validado: la calidad de sueño noche-a-noche es ~impredecible
     (lag-1 autocorr ≈ 0), pero el riesgo de la persona es estable en el tiempo.
  3. Entrena un Random Forest que clasifica el NIVEL DE RIESGO de la persona.
  4. Extrae feature importance (el factor conductual/fisiológico predominante).
  5. Evalúa con PR-AUC / ROC-AUC / Precisión / Recall / F1 contra un baseline
     trivial (siempre predice la clase mayoritaria) — NUNCA con accuracy pelada,
     porque con clases desbalanceadas (~25% duerme mal) un modelo inútil ya saca
     75% de accuracy.
  6. Genera el payload JSON estructurado que C2 le entregaría a C3 (el LLM) por
     estudiante: {risk_level, risk_probability, top_risk_factor, ranking completo}.

IMPORTANTE — limitaciones honestas de este PoC (léelas antes de citar resultados):
  - LifeSnaps NO tiene PSQI. La etiqueta de riesgo aquí es un PROXY construido
    desde el sleep score / eficiencia de sueño de Fitbit (ver build_risk_label).
    Cuando tengas los datos reales de la Capa A (encuesta PSQI, n≥50), hay que
    reemplazar build_risk_label() por el Global PSQI Score (≥5 = mala calidad)
    y re-correr todo. Los números que salgan HOY son para validar que el
    pipeline (C2→C3) funciona, no para citar en el TI como resultado final.
  - LifeSnaps no incluye tiempo de pantalla ni GPS (son variables conductuales
    digitales que solo va a capturar tu app vía UsageStatsManager). Como proxy
    de comportamiento se usa aquí `minutesSedentary` (inactividad) — cuando
    tengas datos reales de la app, agrégalos en FEATURE_COLUMNS_BEHAVIORAL.
  - n=71 es chico. Por eso se usa Repeated Stratified K-Fold (20 repeticiones
    de 5-Fold, no un solo split ni un solo K-Fold) — se comprobó que un solo
    5-Fold con esta n varía muchísimo según la semilla (F1 osciló entre 0.29
    y 0.61 en 20 corridas de prueba). Las métricas que imprime el script son
    mean ± std sobre esas 20 repeticiones, no un solo número.

Cómo correrlo con datos reales:
  1. Descarga daily_fitbit_sema_df_unprocessed.csv de LifeSnaps:
     - Fuente oficial: https://zenodo.org/records/7229547
     - Mirror en Kaggle: https://www.kaggle.com/datasets/skywescar/lifesnaps-fitbit-dataset
  2. Ponlo en esta misma carpeta (o cambia DATA_PATH abajo).
  3. Cambia USE_SYNTHETIC_DATA = False.
  4. python3 poc_c2_random_forest.py
"""

import json
import sys
import warnings
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # consola de Windows por defecto usa cp1252, truena con tildes/─
except AttributeError:
    pass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────
# CONFIG — lo primero que tienes que tocar
# ──────────────────────────────────────────────────────────────────────────
USE_SYNTHETIC_DATA = False  # ← ya tenemos el CSV real de LifeSnaps
DATA_PATH = Path("rais_anonymized/rais_anonymized/csv_rais_anonymized/daily_fitbit_sema_df_unprocessed.csv")
RANDOM_STATE = 42
N_FOLDS = 5                 # Stratified K-Fold (con n≈71, 5 folds ≈ 14 personas por fold)
N_REPEATS = 20               # repite el 5-Fold con 20 semillas distintas y promedia — con n=71
                              # un solo 5-Fold (una sola asignación de qué persona cae en qué
                              # fold) resultó tener varianza enorme: se probó y el F1 de UN
                              # solo run varió entre 0.29 y 0.61 según la semilla. Reportar
                              # mean±std de 20 repeticiones es lo honesto, no un solo número.
RISK_LABEL_QUANTILE = 0.25  # Bottom 25% de calidad de sueño = "riesgo alto" (proxy de ~25% que "duerme mal", igual que en tu comparación de modelos ya corrida)

# Columnas fisiológicas y conductuales que existen en LifeSnaps (schema real
# del CSV daily_fitbit_sema_df_unprocessed.csv). Si tu CSV real trae otros
# nombres de columna, ajústalos aquí — el resto del script no necesita tocarse.
COL_ID = "id"
COL_DATE = "date"

FEATURE_COLUMNS_PHYSIO = [
    "rmssd",                    # HRV
    "nremhr",                   # HR durante sueño no-REM
    "resting_hr",                # HR en reposo (menos missing que nremhr, se deja ambas)
    "spo2",                     # saturación de oxígeno
    "full_sleep_breathing_rate",
    "minutesToFallAsleep",      # latencia de sueño
    "minutesAsleep",            # duración de sueño
    "minutesAwake",
    "sleep_efficiency",         # eficiencia de sueño real del export (no derivada)
    "sleep_points_percentage",  # score de sueño de Fitbit (0-1)
]
FEATURE_COLUMNS_BEHAVIORAL = [
    "steps",
    "sedentary_minutes",        # nombre real de la columna en LifeSnaps — proxy de inactividad, NO es tiempo de pantalla real
]
ALL_RAW_COLUMNS = FEATURE_COLUMNS_PHYSIO + FEATURE_COLUMNS_BEHAVIORAL

# NOTA — se probó agregar STAI (ansiedad) y PANAS (afecto) del export de
# encuestas de LifeSnaps (RAIS) como features psicológicas, agregadas a nivel
# de persona igual que el resto. Son una fuente 100% independiente del
# wearable, así que no hay riesgo de repetir la fuga de datos ya documentada.
# Con un solo Stratified 5-Fold (semilla fija) parecían mejorar mucho el F1,
# pero bajo Repeated Stratified K-Fold (20 semillas) esa mejora no se sostuvo:
# F1 quedó estadísticamente igual (0.490 vs 0.485) y ROC-AUC/PR-AUC bajaron
# levemente (0.705 vs 0.733 / 0.462 vs 0.492) — no una mejora real, era ruido
# de esa semilla puntual. Se descartaron del feature set por eso, no por
# error de fuga. Si más adelante mejora la cobertura de encuestas (hoy solo
# 53-51/71 personas las contestaron), vale la pena volver a intentarlo.


# ──────────────────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ──────────────────────────────────────────────────────────────────────────
def make_synthetic_data(n_people=71, min_days=14, max_days=60, seed=RANDOM_STATE):
    """
    Genera datos sintéticos con la MISMA forma/columnas que LifeSnaps, solo
    para poder correr y validar el pipeline sin depender de la descarga real.
    NO USAR estos números para nada que se cite en el TI — es solo plomería.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for person_idx in range(n_people):
        person_id = f"synthetic_{person_idx:03d}"
        # Cada persona tiene un "nivel de riesgo latente" estable (esto es
        # justamente lo que la PoC de autocorrelación ya demostró: hay un
        # rasgo estable de persona, no de noche)
        latent_risk = rng.beta(2, 5)  # sesgado hacia bajo riesgo, con cola de alto riesgo
        n_days = rng.integers(min_days, max_days)
        for _ in range(n_days):
            sleep_efficiency = np.clip(rng.normal(0.90 - 0.25 * latent_risk, 0.05), 0.4, 1.0)
            rows.append({
                COL_ID: person_id,
                COL_DATE: "synthetic",
                "rmssd": max(5, rng.normal(45 - 15 * latent_risk, 8)),
                "nremhr": rng.normal(62 + 8 * latent_risk, 5),
                "spo2": np.clip(rng.normal(96.5 - 1.5 * latent_risk, 0.8), 88, 100),
                "full_sleep_breathing_rate": rng.normal(15 + latent_risk, 1.2),
                "minutesToFallAsleep": max(0, rng.normal(15 + 25 * latent_risk, 8)),
                "minutesAsleep": max(120, rng.normal(430 - 80 * latent_risk, 40)),
                "minutesAwake": max(0, rng.normal(20 + 30 * latent_risk, 10)),
                "sleep_points_percentage": sleep_efficiency,
                "steps": max(0, rng.normal(8000 - 3000 * latent_risk, 1800)),
                "minutesSedentary": max(0, rng.normal(500 + 150 * latent_risk, 80)),
            })
    return pd.DataFrame(rows)


def load_data():
    if USE_SYNTHETIC_DATA:
        print("[i] Usando datos SINTÉTICOS (solo para validar el pipeline).")
        return make_synthetic_data()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No encuentro {DATA_PATH}. Descarga daily_fitbit_sema_df_unprocessed.csv "
            "de LifeSnaps (ver docstring del script) o deja USE_SYNTHETIC_DATA=True."
        )
    df = pd.read_csv(DATA_PATH)

    # El export de LifeSnaps trae varias columnas como string por participantes
    # con valores faltantes ('nan' literal, etc.) — forzamos numérico y dejamos
    # NaN donde no se pueda parsear, en vez de que truene el script.
    for col in ALL_RAW_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            print(f"[!] Aviso: columna '{col}' no está en el CSV real — se omite esa feature.")

    return df


# ──────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING — agregación a NIVEL DE PERSONA
# ──────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    present_cols = [c for c in ALL_RAW_COLUMNS if c in df.columns]

    agg = df.groupby(COL_ID)[present_cols].agg(["mean", "std"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()

    # Consistencia inter-noche explícita (una de las variables fisiológicas
    # pedidas en el diseño): qué tan variable es la duración de sueño de esa
    # persona noche a noche. std alto = sueño inconsistente.
    if "minutesAsleep_std" in agg.columns:
        agg = agg.rename(columns={"minutesAsleep_std": "sleep_duration_inconsistency"})

    agg = agg.dropna(subset=[c for c in agg.columns if c != COL_ID], how="all")
    # Para features individuales con NaN aislado (persona sin std por 1 solo
    # día de datos, etc.), imputamos con la mediana de la población.
    feature_cols = [c for c in agg.columns if c != COL_ID]
    agg[feature_cols] = agg[feature_cols].fillna(agg[feature_cols].median())

    return agg, feature_cols


# ──────────────────────────────────────────────────────────────────────────
# 3. ETIQUETA DE RIESGO (proxy — ver limitaciones en el docstring del módulo)
# ──────────────────────────────────────────────────────────────────────────
def build_risk_label(person_df: pd.DataFrame):
    """
    Devuelve (y, columna_usada_para_la_etiqueta).

    OJO CON FUGA DE DATOS (data leakage): la columna que uses aquí para
    construir 'y' NO puede quedarse también como feature de entrada — si no,
    el modelo "predice" el riesgo mirando literalmente el número del que
    salió la etiqueta, y las métricas van a salir artificialmente perfectas
    (esto es precisamente lo que le pasaría a este PoC con
    sleep_points_percentage si no la excluyéramos de X en main()).

    En el sistema real esto no es un problema: la etiqueta vendrá del PSQI
    (Capa A, encuesta independiente) y las features vendrán 100% del
    wearable — dos fuentes distintas, sin fuga posible.
    """
    quality_col = "sleep_points_percentage_mean"
    if quality_col not in person_df.columns:
        raise ValueError(
            "No encontré 'sleep_points_percentage' para construir la etiqueta proxy. "
            "Define aquí qué columna de tu dataset real representa calidad/eficiencia "
            "de sueño, o conecta directamente el PSQI Global Score de la Capa A "
            "(≥5 = riesgo) cuando lo tengas."
        )
    threshold = person_df[quality_col].quantile(RISK_LABEL_QUANTILE)
    # Riesgo alto = 1 si su calidad de sueño promedio está en el cuartil más bajo
    y = (person_df[quality_col] <= threshold).astype(int)
    raw_source_name = quality_col.rsplit("_", 1)[0]  # "sleep_points_percentage_mean" -> "sleep_points_percentage"
    return y, raw_source_name


# ──────────────────────────────────────────────────────────────────────────
# 4. ENTRENAMIENTO + EVALUACIÓN (Repeated Stratified K-Fold — no un solo split)
# ──────────────────────────────────────────────────────────────────────────
def make_rf(seed):
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=5,          # n chica → limitar profundidad para no sobreajustar
        class_weight="balanced",
        random_state=seed,
    )


def train_evaluate(X: pd.DataFrame, y: pd.Series, n_repeats: int = N_REPEATS):
    """
    Repite el Stratified K-Fold con N_REPEATS semillas distintas y reporta
    mean±std de cada métrica, en vez de un solo número de una sola corrida.

    Se probó esto porque un solo Stratified 5-Fold (una sola asignación de
    qué persona cae en qué fold) resultó tener varianza enorme con n=71: el
    F1 de una corrida osciló entre 0.29 y 0.61 según la semilla nada más.
    Reportar un solo run (aunque sea K-Fold) es tan engañoso con esta n como
    reportar accuracy pelada con clases desbalanceadas — mismo espíritu que
    ya aplicamos ahí, aplicado a la varianza del split.
    """
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
    rf_stats = {
        f"{m}_mean": float(np.mean([r[m] for r in per_repeat])) for m in metric_names
    }
    rf_stats.update({
        f"{m}_std": float(np.std([r[m] for r in per_repeat])) for m in metric_names
    })

    results = {
        "random_forest": rf_stats,
        "baseline_trivial": metrics(y, baseline_pred),
        "prevalencia_clase_riesgo": float(y.mean()),
        "n_repeats": n_repeats,
    }

    # Probabilidad promedio de riesgo entre las N_REPEATS corridas — más
    # estable para el payload C2→C3 que la de una sola corrida (un
    # estudiante límite puede saltar de 0.45 a 0.55 solo por la semilla).
    mean_risk_proba = proba_sum / n_repeats

    # Feature importance promediada sobre N_REPEATS fits en TODO el dataset
    # (la CV de arriba es solo para medir desempeño honesto; esto es para el
    # ranking de factores que efectivamente le llega al LLM) — por la misma
    # razón que las métricas, un solo fit puede dar un ranking distinto según
    # la semilla con esta n.
    importances_sum = np.zeros(X.shape[1])
    for r in range(n_repeats):
        rf_full = make_rf(RANDOM_STATE + r)
        rf_full.fit(X, y)
        importances_sum += rf_full.feature_importances_
    mean_importances = pd.Series(importances_sum / n_repeats, index=X.columns)

    return results, mean_importances, mean_risk_proba


# ──────────────────────────────────────────────────────────────────────────
# 5. PAYLOAD ESTRUCTURADO C2 → C3 (esto es lo que le llega al LLM)
# ──────────────────────────────────────────────────────────────────────────
def build_llm_payloads(person_df, feature_cols, importances, risk_proba, y_true):
    importances_sorted = importances.sort_values(ascending=False)

    payloads = []
    for i, row in person_df.reset_index(drop=True).iterrows():
        top_factor = importances_sorted.index[0]
        payloads.append({
            "student_id": row[COL_ID],
            "risk_level": "alto" if risk_proba[i] >= 0.5 else "bajo",
            "risk_probability": round(float(risk_proba[i]), 3),
            "top_risk_factor": top_factor,
            "top_risk_factor_value": round(float(row[top_factor]), 2),
            "feature_importance_ranked": [
                {"feature": f, "importance": round(float(v), 4)}
                for f, v in importances_sorted.items()
            ],
        })
    return payloads


# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────
def main():
    raw_df = load_data()
    person_df, feature_cols = engineer_features(raw_df)
    y, label_source_raw = build_risk_label(person_df)

    # Excluimos TODAS las variantes (_mean, _std, etc.) de la columna cruda
    # usada para construir la etiqueta — no solo la exacta. Dejar el _std
    # cuando ya excluiste el _mean sigue siendo fuga de datos, porque ambas
    # vienen de la misma medición.
    feature_cols = [c for c in feature_cols if not c.startswith(label_source_raw + "_")]
    X = person_df[feature_cols]

    print(f"\n[i] n personas = {len(person_df)} | features = {len(feature_cols)}")
    print(f"[i] Prevalencia de 'riesgo alto' (proxy): {y.mean():.1%}\n")

    results, importances, risk_proba = train_evaluate(X, y)

    print("─" * 60)
    print(f"RESULTADOS (Repeated Stratified {N_FOLDS}-Fold CV, {N_REPEATS} repeticiones)")
    print("─" * 60)
    print("\nrandom_forest (mean ± std sobre {} repeticiones):".format(N_REPEATS))
    for metric in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
        mean_v = results["random_forest"][f"{metric}_mean"]
        std_v = results["random_forest"][f"{metric}_std"]
        print(f"   {metric:10s}: {mean_v:.3f} ± {std_v:.3f}")
    print("\nbaseline_trivial:")
    for metric, value in results["baseline_trivial"].items():
        print(f"   {metric:10s}: {value:.3f}")

    payloads = build_llm_payloads(person_df, feature_cols, importances, risk_proba, y)

    out_path = Path("c2_output_sample.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payloads[:3], f, indent=2, ensure_ascii=False)

    print(f"\n[i] Muestra de payload C2→C3 (3 estudiantes) guardada en {out_path}")
    print(json.dumps(payloads[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
