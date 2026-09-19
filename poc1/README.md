# PoC — C2 (Random Forest) HapSleep

PoC del motor de ML que clasifica el **nivel de riesgo de mala calidad de sueño de la persona** (no de una noche puntual) y calcula **feature importance**, listo para alimentar al LLM (C3).

## Dataset elegido: LifeSnaps (Fitbit Sense, n=71)

Es el mismo dataset que ya usaron para el análisis de autocorrelación/estabilidad (r=0.727). Se eligió por continuidad metodológica y porque es el único de los dos (SSAQS/LifeSnaps) con variables fisiológicas de wearable reales (HR, HRV, SpO2, sueño) publicado académicamente (Nature *Scientific Data*, 2022) con licencia abierta.

- Oficial (Zenodo): https://zenodo.org/records/7229547
- Mirror Kaggle: https://www.kaggle.com/datasets/skywescar/lifesnaps-fitbit-dataset

Ya corrido con el `daily_fitbit_sema_df_unprocessed.csv` real que subiste (7,410 filas, 71 personas, 64–244 días por persona). El export completo (`rais_anonymized/`) trae además el resto del dataset LifeSnaps — Mongo dump y encuestas (`scored_surveys/`: BREQ, PANAS, personalidad, STAI, TTM). El script apunta a `rais_anonymized/rais_anonymized/csv_rais_anonymized/daily_fitbit_sema_df_unprocessed.csv`; si vuelves a bajar el dataset y lo dejas en otra ruta, ajusta `DATA_PATH`.

## Cómo correrlo

```bash
pip install pandas numpy scikit-learn
python3 poc_c2_random_forest.py
```

Por defecto (`USE_SYNTHETIC_DATA = False`) corre con el CSV real de LifeSnaps que ya está en esta carpeta. Si necesitas validar que el pipeline no tiene bugs sin depender del dataset real, pon `USE_SYNTHETIC_DATA = True` — genera datos sintéticos con la misma forma, pero esos números no sirven para citar en el TI.

## Resultados con datos reales (Repeated Stratified 5-Fold CV, 20 repeticiones, n=71)

| Métrica | Random Forest (mean ± std) | Baseline trivial |
|---|---|---|
| Precisión | 0.531 ± 0.079 | 0.000 |
| Recall | 0.439 ± 0.084 | 0.000 |
| F1 | 0.478 ± 0.077 | 0.000 |
| ROC-AUC | 0.728 ± 0.033 | — |
| PR-AUC | 0.481 ± 0.046 | — |

Top factor de riesgo global: `minutesAwake_std` (inconsistencia de vigilia nocturna noche a noche), seguido de `sleep_duration_inconsistency` (std de `minutesAsleep`) — se mantiene estable entre repeticiones.

**Por qué "mean ± std sobre 20 repeticiones" y no un solo número:** la primera versión de este PoC reportaba un solo Stratified 5-Fold (F1=0.25) y esa cifra parecía calzar sospechosamente bien con la regresión logística ya corrida (0.248) — se leyó como "el techo real del problema". No lo era: se probó correr el mismo 5-Fold con 20 semillas distintas y el F1 de una sola corrida osciló entre **0.29 y 0.61** solo por azar en qué persona cayó en qué fold. Con n=71, un solo K-Fold sigue siendo una sola muestra de la varianza — el 0.25 original fue una corrida particularmente desafortunada, no el desempeño real del modelo. La cifra honesta es el promedio de muchas repeticiones: **F1 ≈ 0.48 ± 0.08**.

## Limitaciones honestas (no te las saltes)

1. **Etiqueta de riesgo = proxy.** LifeSnaps no tiene PSQI. Se usa el sleep score de Fitbit (`sleep_points_percentage`) bottom-25% como proxy de "riesgo alto". Cuando tengas los datos de la Capa A (encuesta PSQI, n≥50), reemplaza `build_risk_label()` por el Global PSQI Score (≥5 = mala calidad) y vuelve a correr TODO — los números de hoy no van al TI como resultado final.
2. **Faltan variables conductuales digitales reales.** LifeSnaps no tiene tiempo de pantalla ni GPS (eso solo lo captura tu app vía `UsageStatsManager`). Se usa `sedentary_minutes` como proxy temporal.
3. **n=71 es chico y con mucho missing data real**: spo2 falta en 82.9% de las filas, sleep_points_percentage en 74.7%, rmssd/nremhr en 66.6%. Se imputa con mediana poblacional a nivel de feature ya agregada — no es ideal, pero evita que el pipeline truene; documenta esto si citas resultados.
4. **n=71 es chico para modelar.** Por eso se evalúa con Repeated Stratified 5-Fold CV (20 repeticiones), no con un solo train/test split ni un solo K-Fold — ver la sección de resultados arriba para el porqué.
5. **Dos fugas de datos reales se encontraron y corrigieron durante la construcción de este PoC** (no en una revisión externa — las encontré yo mismo corriendo el script, por eso las dejo documentadas):
   - La primera corrida dejó `sleep_points_percentage_mean` (la fuente de la etiqueta) también como feature → ROC-AUC 0.994 con datos sintéticos, obviamente circular.
   - La segunda corrida excluyó el `_mean` pero dejó el `_std` de la misma columna cruda → con datos REALES esto igual infló el resultado a ROC-AUC 0.955 / F1 0.714, con `sleep_points_percentage_std` como el feature más importante (0.25). Se corrigió excluyendo TODAS las variantes (`_mean`, `_std`) de la columna cruda usada para la etiqueta, no solo la exacta. La tabla de arriba ya es la versión corregida.
   - Moraleja para cuando conectes el PSQI real: como el PSQI es una fuente 100% independiente del wearable, esta fuga específica no se puede repetir — pero vale la pena correr el mismo chequeo (¿alguna feature de entrada deriva de la misma medición que la etiqueta?) cada vez que cambies el origen de la etiqueta.
6. **Se probó agregar STAI (ansiedad) y PANAS (afecto) como features psicológicas** — están en el export de encuestas de LifeSnaps (`scored_surveys/`), son una fuente 100% independiente del wearable (sin riesgo de repetir la fuga del punto 5), y se agregaron a nivel de persona igual que el resto. Con un solo Stratified 5-Fold (semilla fija) parecían mejorar mucho el F1 (0.25→0.39) — pero bajo Repeated Stratified K-Fold esa "mejora" no se sostuvo: F1 quedó estadísticamente igual y ROC-AUC/PR-AUC bajaron levemente. Era ruido de esa semilla puntual, no una mejora real. Se descartaron del feature set final por eso (no por fuga de datos) y no están en el script; solo 53/71 y 51/71 personas contestaron STAI/PANAS respectivamente, así que si mejora la cobertura de encuestas vale la pena reintentarlo.

## Contexto de benchmark en la literatura

*Nota de procedencia: este resumen viene de una búsqueda asistida (NotebookLM) sobre los PDFs ya cargados en el notebook de la tesis. Verifica cifras y citas exactas contra el PDF original antes de usarlas en el TI — esto documenta la conclusión y el razonamiento, no reemplaza la lectura de la fuente primaria.*

Antes de poner el F1=0.478±0.077 / ROC-AUC=0.728±0.033 de este PoC al lado de un número de la literatura, se buscó un benchmark comparable: n≈70-100 universitarios, riesgo de sueño a nivel de persona (no por época), Random Forest o similar. No existe uno así en la literatura revisada.

| Estudio | n | Métrica reportada | Valor | Tarea | Por qué no es comparable |
|---|---|---|---|---|---|
| Li et al. (2026) | 242 | ROC-AUC | 0.92 | Riesgo de trastorno de sueño, regresión logística | Adultos mayores preoperatorios (edad media 67.4), no universitarios; regresión logística, no RF |
| Zhang, Yang & Li (2026) | 328 | Accuracy | 84.7% | Clasificación 3-tier de respuesta a intervención, RF | Universitarios y RF, pero sin F1/ROC-AUC/matriz de confusión — no hay con qué comparar en los mismos términos |
| Lee et al. (2023) | 75 | Macro F1 | 0.26–0.69 | 4 etapas de sueño época por época vs. PSG | Tarea distinta: época por época, no riesgo por persona |
| Robbins et al. (2024) | 35 | Accuracy / Sensibilidad | 91–93% / ≥95% | Sueño/vigilia época por época vs. PSG | Accuracy inflada (85–90% de la noche es sueño); tarea distinta |
| Herberger et al. (2025) | 43 | Accuracy / Kappa | 85.0% (κ=0.43) | Sueño/vigilia época por época, anillos en clínica | Especificidad a vigilia <46%; accuracy cruda esconde el desbalance |
| Kim et al. (2025) | 20 | Accuracy / Sens / Spec | 87% / 91% / 83% | Apnea/hipopnea por evento, CNN | Tarea distinta; sin F1/ROC-AUC |
| Aziz et al. (2025) — revisión de 46 estudios | — | % de estudios que reporta F1 | 34.8% (16/46) | Revisión sistemática de IA en wearables | Confirma que la mayoría de la literatura publica accuracy cruda, no F1/ROC-AUC |

**Conclusión honesta:** no hay una cifra de la literatura contra la cual poner el F1/ROC-AUC de este PoC en la misma tabla sin advertir el desajuste de tarea y de población:

1. **Task mismatch** — los estudios con n parecido al de este PoC (35–75) evalúan concordancia época por época contra polisomnografía, no una etiqueta de riesgo por persona.
2. **Population mismatch** — el único ROC-AUC alto de la lista (0.92, Li et al. 2026) es de adultos mayores preoperatorios con regresión logística, no universitarios con RF.
3. **Sin cifra equivalente** — el único estudio con universitarios y RF (Zhang et al. 2026, n=328) reporta solo accuracy cruda, sin F1/ROC-AUC.
4. **Es un patrón, no una excepción** — Aziz et al. (2025) muestra que esto es sistémico: menos de 35% de 46 estudios de IA en wearables reportan F1-Score.

Por eso este PoC no presenta una tabla "mi modelo vs. la literatura" — el hallazgo es justamente que la práctica dominante (accuracy cruda, tareas de época) no es compatible metodológicamente con evaluar riesgo por persona en universitarios con clases desbalanceadas.

## Validación adicional (post-hoc)

### Comparación LR / Random Forest / XGBoost / SVM

`poc_c2_model_comparison.py` evalúa los cuatro modelos con el mismo pipeline sin fuga y la misma metodología (Repeated Stratified 5-Fold, 20 repeticiones) que el modelo oficial de C2 — para que sea una comparación justa y no se mezcle un número de una sola corrida (como el F1=0.248 de LR que se había corrido antes en otro análisis) con el RF ya promediado.

SVM se agregó después de preguntarle a la literatura de la tesis (vía NotebookLM, con las fuentes cargadas del proyecto) si Deep Learning era apropiado dado n=71-150. Respuesta con cita de fuente: ninguna fuente usa o recomienda Deep Learning para clasificación de riesgo sobre datos tabulares agregados por persona con n<200 (los usos de DL en las fuentes son sobre señales crudas época-por-época — otra tarea, ver Kim et al. 2025); en cambio, Aziz et al. (2025), en su revisión de 46 estudios de IA en wearables, ubica a SVM como el tercer algoritmo más usado (26.1%), justo detrás de Random Forest (30.4%) — el modelo clásico "pendiente" que la literatura del dominio sí respalda. Prompt completo y respuesta en `poc_clean/POC1/README.md`.

| Modelo | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| Logistic Regression | 0.462 ± 0.078 | 0.669 ± 0.051 | 0.428 ± 0.059 |
| **Random Forest (elegido)** | **0.478 ± 0.077** | 0.728 ± 0.033 | 0.481 ± 0.046 |
| XGBoost | 0.414 ± 0.077 | 0.755 ± 0.037 | 0.493 ± 0.049 |
| SVM (RBF, `probability=True`) | 0.088 ± 0.092 | 0.668 ± 0.039 | 0.397 ± 0.046 |

Ningún modelo domina en las tres métricas: RF tiene el mejor F1 (aunque casi empatado con LR, dentro de 1 std), XGBoost tiene mejor ROC-AUC/PR-AUC pero el peor recall (0.356) de los tres. Esto refuerza — no contradice — la decisión ya cerrada de usar RF por interpretabilidad: no se sacrifica desempeño relevante por elegir el modelo que además es más fácil de explicarle al LLM (C3) y al usuario final.

**SVM, a pesar de estar respaldado por la literatura, da un F1 muy bajo aquí (0.088)** pese a un ROC-AUC razonable (0.668, comparable al de LR) — la causa más probable es que `predict_proba` de `SVC` se calibra internamente con su propio 5-Fold (Platt scaling), y con una clase minoritaria de ~18 personas en n=71, esa calibración interna es inestable y termina empujando casi todas las probabilidades por debajo de 0.5. No se "arregló" ajustando el umbral de decisión para hacer ver mejor a SVM — se reporta tal como salió, con el mismo umbral 0.5 que los otros tres modelos, porque el punto de este PoC es la comparación justa, no maximizar el número de cada modelo.

### Gini importance vs. Permutation importance

Se comparó el ranking de feature importance que usa el payload C2→C3 (Gini/MDI, el `feature_importances_` nativo de sklearn) contra permutation importance (scoring=ROC-AUC, promediado sobre 20 semillas × 10 permutaciones cada una), porque Gini importance está sesgada a favor de variables continuas de alta cardinalidad y puede no coincidir con qué feature realmente mueve la predicción al perturbarla.

**El top-1 SÍ cambia** entre métodos: Gini pone `minutesAwake_std` primero y `sedentary_minutes_mean` tercero; permutation invierte ese orden. Pero los valores de permutation importance son muy pequeños y están muy cerca entre sí (0.0015 vs. 0.0010 vs. 0.0008 sobre ROC-AUC) — la diferencia entre el 1º y el 3º lugar es marginal, así que el "ganador" exacto es frágil con cualquiera de los dos métodos, no solo con uno.

Lo que sí es estable en ambos rankings es el mismo grupo de arriba: `minutesAwake_std`, `sedentary_minutes_mean`, `minutesAsleep_mean` y `sleep_duration_inconsistency` — inconsistencia de sueño/vigilia noche a noche, e inactividad. **Recomendación para el campo `top_risk_factor` del payload C2→C3:** en vez de reportar un solo "factor ganador" (que puede cambiar de persona a persona por una diferencia de milésimas), considera que C3 (el LLM) reciba el ranking completo — que ya se le manda en `feature_importance_ranked` — y hable de "los factores principales" (plural) en vez de un único factor determinante. El script no se modificó para forzar esto porque es una decisión de diseño del payload, no un bug.

## Output

`c2_output_sample.json` — el payload estructurado que C2 le entregaría a C3 (LLM), por estudiante: `risk_level`, `risk_probability`, `top_risk_factor`, y el ranking completo de feature importance.
