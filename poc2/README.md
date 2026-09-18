# PoC 2 — ML1 (Random Forest) con datos sintéticos de wearable

Complementa a [poc1](../poc1/) (que corre el mismo tipo de pipeline sobre datos reales de LifeSnaps, n=71). Este PoC genera datos de wearable **100% sintéticos** con un riesgo verdadero (`latent_risk`) conocido por persona, y corre el mismo pipeline de ML1 para validar que la arquitectura (agregación por persona → features → Random Forest → evaluación) recupera esa señal cuando existe.

## Por qué esto, si ya está poc1 con datos reales

Con datos reales nunca sabes si el modelo está capturando la señal correcta o solo ruido que casualmente correlaciona con la etiqueta proxy. Con datos sintéticos **sí sabes la verdad**, porque tú mismo la inyectaste al generar los datos — eso permite una validación imposible con datos reales: comparar la predicción del modelo contra el riesgo real, no solo contra una etiqueta derivada (proxy).

**Esto es una validación de pipeline, no una medición de desempeño esperado.** Los números de este PoC siempre van a verse mejor que los de poc1, porque el generador inyecta una relación limpia (con ruido controlado) entre features y riesgo — la vida real tiene missing data, medición imperfecta, y factores que el generador no modela. No cites estos números en el TI como "el modelo funciona con X% de desempeño" — sirven para decir "el pipeline está bien construido: cuando la señal existe, la encuentra."

## Cómo correrlo

```bash
pip install pandas numpy scikit-learn scipy
python generate_synthetic_wearable_data.py   # genera synthetic_wearable_data.csv + synthetic_ground_truth.csv
python poc_ml1_random_forest.py              # corre ML1 sobre esos datos y valida contra el ground truth
```

## Cómo se generan los datos sintéticos

`generate_synthetic_wearable_data.py`: 150 personas sintéticas (más que las 71 reales de LifeSnaps), cada una con un `latent_risk ~ Beta(2, 5)` fijo (0 = bajo riesgo, 1 = alto riesgo, sesgado hacia bajo riesgo con cola de alto riesgo) — replica el hallazgo ya validado de que el riesgo es un rasgo **estable de la persona**, no de la noche. Entre 30 y 120 días de datos por persona, con las mismas columnas fisiológicas y conductuales que LifeSnaps (HRV, HR, SpO2, sueño, pasos, sedentarismo).

**Detalle importante:** `minutesAwake` y `sedentary_minutes` no solo suben de nivel con el riesgo — también se vuelven más **inconsistentes noche a noche** a mayor riesgo (la desviación estándar del generador escala con `latent_risk`). Esto replica a propósito el hallazgo real de poc1, donde `minutesAwake_std` / `sedentary_minutes_mean` salieron como los factores más importantes. Si ML1 corre sobre este dataset sintético y encuentra esos mismos factores arriba, es evidencia de que el pipeline funciona como se espera.

`latent_risk` se guarda en `synthetic_ground_truth.csv` y **nunca se le pasa al modelo** — solo se usa después de entrenar, para validar.

## Resultados (Repeated Stratified 5-Fold CV, 20 repeticiones, n=150 sintéticas)

| Métrica | Random Forest (mean ± std) | Baseline trivial |
|---|---|---|
| Precisión | 0.837 ± 0.021 | 0.000 |
| Recall | 0.904 ± 0.027 | 0.000 |
| F1 | 0.869 ± 0.019 | 0.000 |
| ROC-AUC | 0.988 ± 0.002 | — |
| PR-AUC | 0.967 ± 0.005 | — |

Como se esperaba, muchísimo más alto que en poc1 (F1=0.478±0.077 con datos reales) — de nuevo, esto no es "ML1 es mejor de lo que pensábamos", es que los datos sintéticos son más limpios que la realidad por construcción.

**Top feature importance (Gini, promedio 20 fits):** `sleep_efficiency_mean`, `minutesAwake_mean`, `sedentary_minutes_mean`, `minutesToFallAsleep_mean`, `minutesAwake_std` — coincide con las variables que el generador usó para inyectar el riesgo, confirmando que el pipeline no está capturando artefactos espurios.

## Validación contra ground truth (lo que no se puede hacer con datos reales)

| Chequeo | Resultado |
|---|---|
| Correlación de Spearman (`risk_probability` del modelo vs. `latent_risk` real) | **0.898** (p<0.0001) |
| ROC-AUC del modelo contra la etiqueta "verdadera" (top 25% de `latent_risk`, no el proxy) | **0.991** |
| Acuerdo entre la etiqueta proxy (`sleep_points_percentage`) y la etiqueta verdadera (`latent_risk`) | 97.3% |

**Lectura:** el modelo recupera casi perfectamente el riesgo real inyectado (correlación 0.898, ROC-AUC 0.991 contra la verdad). El 2.7% de desacuerdo entre la etiqueta proxy y la verdadera muestra que, incluso en un mundo sintético con ruido controlado, un proxy de calidad de sueño (`sleep_points_percentage`) no es 100% idéntico al riesgo subyacente — un recordatorio de que en poc1, donde el proxy es la única etiqueta disponible (no hay PSQI), ese desacuerdo real es casi con certeza mayor a 2.7%.

## Relación con poc1

| | poc1 | poc2 |
|---|---|---|
| Datos | Reales (LifeSnaps, n=71) | Sintéticos (n=150, generados aquí) |
| Etiqueta | Proxy (`sleep_points_percentage`), única disponible | Proxy + ground truth conocido (`latent_risk`) |
| Qué valida | Desempeño real esperado, honesto | Que la arquitectura del pipeline funciona cuando la señal existe |
| F1 (Repeated Stratified 5-Fold) | 0.478 ± 0.077 | 0.869 ± 0.019 (no comparable directamente — ver arriba) |

## Pendiente / próximos pasos

- Si se quiere estresar el pipeline más (no solo confirmarlo), se puede repetir este PoC con ruido de medición más agresivo, missing data inyectado, o un `latent_risk` que no correlacione linealmente con las features — para ver en qué punto el pipeline deja de recuperar la señal.
- Este PoC es sobre **ML1** (clasificación de riesgo). El LLM (recomendaciones a partir del output de ML1) queda para otro PoC aparte, con su propio diseño de datos sintéticos (payloads tipo `c2_output_sample.json`, no datos de wearable crudos).
