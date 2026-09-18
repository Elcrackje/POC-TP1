"""
PoC 2 — Generador de datos sintéticos de wearable para ML1
=============================================================

Genera un dataset sintético con la MISMA forma/columnas fisiológicas y
conductuales que LifeSnaps (el dataset real usado en poc1), pero con un
"riesgo verdadero" (latent_risk) conocido y guardado aparte en
synthetic_ground_truth.csv.

Por qué esto es útil (y distinto de simplemente "más de lo mismo" que
poc1): con datos reales nunca sabes si el modelo está capturando la señal
correcta o solo ruido que casualmente correlaciona con la etiqueta proxy.
Con datos sintéticos SÍ sabes la verdad, porque tú mismo la inyectaste —
así que puedes validar si ML1 (Random Forest) recupera el riesgo real
(latent_risk), no solo si acierta contra una etiqueta derivada.

Estructura del generador (por persona):
  - Cada persona tiene un latent_risk ~ Beta(2, 5) fijo (sesgado a bajo
    riesgo, con cola de alto riesgo) — replica el hallazgo ya validado en
    la tesis: el riesgo es un rasgo ESTABLE de la persona, no de la noche.
  - Cada noche/día se generan métricas fisiológicas y conductuales
    correlacionadas con ese latent_risk + ruido.
  - CLAVE: minutesAwake y sedentary_minutes no solo suben de nivel con el
    riesgo, también se vuelven más INCONSISTENTES noche a noche (mayor
    varianza) — esto replica el hallazgo real de poc1, donde
    minutesAwake_std / sedentary_minutes_mean salieron como los factores
    más importantes. Si el pipeline de ML1 corre sobre este dataset
    sintético y encuentra esos mismos factores arriba, es evidencia de que
    el pipeline funciona como se espera, no que encontró ruido.

Output:
  - synthetic_wearable_data.csv   — igual forma que daily_fitbit_sema_df_unprocessed.csv de LifeSnaps
  - synthetic_ground_truth.csv    — id, latent_risk (el riesgo real, NUNCA se le pasa al modelo)
"""
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_PEOPLE = 150       # más que los 71 reales de LifeSnaps -- para ver cómo se comporta ML1 con más n
MIN_DAYS = 30
MAX_DAYS = 120

COL_ID = "id"
COL_DATE = "date"


def make_synthetic_wearable_data(n_people=N_PEOPLE, min_days=MIN_DAYS, max_days=MAX_DAYS, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    rows = []
    ground_truth = []

    for person_idx in range(n_people):
        person_id = f"synthetic_{person_idx:03d}"
        latent_risk = rng.beta(2, 5)  # 0 = bajo riesgo, 1 = alto riesgo -- rasgo fijo de la persona
        ground_truth.append({COL_ID: person_id, "latent_risk": latent_risk})

        n_days = rng.integers(min_days, max_days)
        # Inconsistencia noche a noche que ESCALA con el riesgo -- no solo el nivel promedio sube,
        # la varianza también. Esto es lo que hace que el "_std" de estas columnas sea predictivo,
        # igual que se encontró con datos reales en poc1.
        awake_std_for_person = 8 + 20 * latent_risk
        sedentary_std_for_person = 40 + 60 * latent_risk

        for _ in range(n_days):
            minutes_asleep = max(120, rng.normal(430 - 80 * latent_risk, 40))
            minutes_awake = max(0, rng.normal(20 + 35 * latent_risk, awake_std_for_person))
            sedentary = max(0, rng.normal(450 + 200 * latent_risk, sedentary_std_for_person))

            rows.append({
                COL_ID: person_id,
                COL_DATE: f"day_{_}",
                "rmssd": max(5, rng.normal(45 - 15 * latent_risk, 8)),
                "nremhr": rng.normal(62 + 8 * latent_risk, 5),
                "resting_hr": rng.normal(65 + 6 * latent_risk, 6),
                "spo2": np.clip(rng.normal(96.5 - 1.5 * latent_risk, 0.8), 88, 100),
                "full_sleep_breathing_rate": rng.normal(15 + latent_risk, 1.2),
                "minutesToFallAsleep": max(0, rng.normal(15 + 25 * latent_risk, 8)),
                "minutesAsleep": minutes_asleep,
                "minutesAwake": minutes_awake,
                "sleep_efficiency": np.clip(rng.normal(0.90 - 0.20 * latent_risk, 0.05), 0.4, 1.0),
                "sleep_points_percentage": np.clip(rng.normal(0.85 - 0.30 * latent_risk, 0.07), 0.3, 1.0),
                "steps": max(0, rng.normal(8500 - 3000 * latent_risk, 1800)),
                "sedentary_minutes": sedentary,
            })

    return pd.DataFrame(rows), pd.DataFrame(ground_truth)


if __name__ == "__main__":
    df, gt = make_synthetic_wearable_data()
    df.to_csv("synthetic_wearable_data.csv", index=False)
    gt.to_csv("synthetic_ground_truth.csv", index=False)
    print(f"[i] {len(gt)} personas sinteticas, {len(df)} filas dia-persona")
    print(f"[i] Guardado: synthetic_wearable_data.csv, synthetic_ground_truth.csv")
    print(f"[i] Prevalencia real 'riesgo alto' (latent_risk >= percentil 75): "
          f"{(gt['latent_risk'] >= gt['latent_risk'].quantile(0.75)).mean():.1%}")
