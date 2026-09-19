# Findings — PoC 1 (HapSleep C2, Random Forest)

Resumen de lo hecho y encontrado en esta sesión de trabajo sobre el PoC del motor de ML (C2) de HapSleep. Todo lo que sigue se corrió con **datos reales de LifeSnaps (Fitbit Sense, n=71)**, nunca con datos sintéticos — `USE_SYNTHETIC_DATA = False` en todas las corridas citadas aquí. Los datos sintéticos (`make_synthetic_data()` en `poc_c2_random_forest.py`) solo existen como plomería para probar que el pipeline no truena si algún día no tienes el CSV real a mano; ningún número de este documento sale de ahí.

## Qué se hizo (todo sobre datos reales)

1. **Se corrigió la ruta al dataset real.** El CSV (`daily_fitbit_sema_df_unprocessed.csv`) ya no estaba en el nivel superior de la carpeta — se había movido dentro de `rais_anonymized/rais_anonymized/csv_rais_anonymized/` al descomprimir el export completo de LifeSnaps. El script apuntaba a la ruta vieja y hubiera fallado.
2. **Se exploró el export completo de LifeSnaps** (`rais_anonymized/`, no solo los dos CSV que ya tenías) — Mongo dump + encuestas puntuadas (BREQ, PANAS, personalidad Big Five, STAI, TTM). Se confirmó que **sigue sin haber PSQI** ahí (la limitación #1 del README no cambia).
3. **Se integró STAI (ansiedad) y PANAS (afecto)** como features psicológicas agregadas a nivel de persona (mean/std), unidas por `user_id`/`id` — fuente 100% independiente del wearable, sin riesgo de repetir las fugas de datos ya documentadas.
4. **Se detectó que un solo Stratified 5-Fold CV es engañoso con n=71.** Se corrió el mismo split 20 veces con semillas distintas (`checks/robustness_check.py`) y el F1 de una sola corrida osciló entre **0.29 y 0.61** solo por azar en qué persona cayó en qué fold.
5. **Se re-evaluó el experimento STAI/PANAS bajo esa metodología robusta** — la "mejora" que parecía haber con una sola semilla (F1 0.25→0.39) no se sostuvo. Se descartaron del feature set final.
6. **Se migró el pipeline oficial (`poc_c2_random_forest.py`) a Repeated Stratified 5-Fold CV** (20 repeticiones), reportando mean±std en vez de un solo número — para F1, ROC-AUC, PR-AUC, precisión, recall, y también para el feature importance y el `risk_probability` del payload C2→C3.
7. **Se calculó permutation importance** (20 semillas × 10 permutaciones, scoring=ROC-AUC) sobre datos reales y se comparó contra el Gini/MDI importance que ya usaba el payload (`checks/permutation_check.py`).
8. **Se corrió una comparación Logistic Regression / Random Forest / XGBoost** (`poc_c2_model_comparison.py`) con el mismo pipeline sin fuga y la misma metodología (Repeated Stratified 5-Fold, 20 repeticiones) para los tres modelos — comparación justa, no mezclada con números viejos de una sola corrida.
9. **Se investigó (vía búsqueda asistida en NotebookLM) si existe un benchmark de la literatura comparable** a este PoC (n≈70-100 universitarios, riesgo por persona, Random Forest) — se documentó por qué no lo hay, con 7 estudios/revisión revisados.
10. **Se redactó** (pendiente de tu revisión, no incorporado al `.docx` del TI) el párrafo de discusión del Capítulo 4 que justifica F1+ROC-AUC sobre accuracy cruda, citando los mismos papers.

## Insights encontrados

1. **El F1=0.25 documentado originalmente no era el "techo real" del problema — era una corrida particularmente desafortunada.** El desempeño honesto y estable del modelo, bajo Repeated Stratified 5-Fold (20 repeticiones), es:

   | Métrica | Random Forest (mean ± std) |
   |---|---|
   | F1 | 0.478 ± 0.077 |
   | ROC-AUC | 0.728 ± 0.033 |
   | PR-AUC | 0.481 ± 0.046 |
   | Precisión | 0.531 ± 0.079 |
   | Recall | 0.439 ± 0.084 |

   Casi el doble del F1 que estaba en el README antes de esta sesión.

2. **STAI/PANAS no mejoran el modelo de forma robusta.** Con una sola semilla parecía una mejora grande; bajo repeated CV, el F1 quedó estadísticamente igual y ROC-AUC/PR-AUC bajaron levemente. Se descartaron del pipeline oficial por falta de beneficio real, no por fuga de datos — el código queda documentado en `checks/robustness_check.py` para que el experimento sea reproducible.

3. **El ranking de "factor de riesgo top" no es tan estable como parece.** Gini importance pone `minutesAwake_std` primero; permutation importance pone `sedentary_minutes_mean` primero. La diferencia entre ambos es marginal en magnitud (0.0015 vs. 0.0010 sobre ROC-AUC) — el top-4 cluster sí es estable: inconsistencia de sueño/vigilia noche a noche (`minutesAwake_std`, `sleep_duration_inconsistency`, `minutesAsleep_mean`) e inactividad (`sedentary_minutes_mean`). **Implicación de diseño:** el payload C2→C3 debería comunicarle a C3 (el LLM) "los factores principales" en plural, no un único ganador — ya se le manda el ranking completo en `feature_importance_ranked`, solo falta que el prompt de C3 lo use así.

4. **Ningún modelo (LR/RF/XGBoost/SVM) domina en todo, y eso favorece la decisión ya tomada de usar RF por interpretabilidad:**

   | Modelo | F1 | ROC-AUC | PR-AUC |
   |---|---|---|---|
   | Logistic Regression | 0.462 ± 0.078 | 0.669 ± 0.051 | 0.428 ± 0.059 |
   | **Random Forest** | **0.478 ± 0.077** | 0.728 ± 0.033 | 0.481 ± 0.046 |
   | XGBoost | 0.414 ± 0.077 | 0.755 ± 0.037 | 0.493 ± 0.049 |
   | SVM | 0.088 ± 0.092 | 0.668 ± 0.039 | 0.397 ± 0.046 |

   RF tiene el mejor F1, XGBoost mejor ROC-AUC/PR-AUC pero el peor recall de los tres (0.356) — usar RF no cuesta desempeño relevante y sí gana interpretabilidad para alimentar a C3. SVM se agregó en una sesión posterior, después de confirmar con la literatura de la tesis (vía NotebookLM) que Deep Learning no está respaldado para n<200 con datos tabulares agregados, y que SVM sí es el modelo clásico que la literatura del dominio sugiere probar (Aziz et al. 2025: 3er algoritmo más usado en 46 estudios de IA en wearables). Su F1 salió muy bajo pese a un ROC-AUC razonable — probablemente por inestabilidad en la calibración interna de `predict_proba` (Platt scaling) con una clase minoritaria tan chica. Se reporta tal cual, sin ajustar el umbral para mejorar el número.

5. **No existe un benchmark de la literatura directamente comparable** a esta tarea (n≈70-100 universitarios, riesgo por persona, RF). Razones documentadas en el README: task mismatch (la mayoría evalúa clasificación época-por-época contra PSG, no riesgo por persona), population mismatch (el único ROC-AUC alto de la literatura revisada es en adultos mayores preoperatorios con regresión logística), y falta de cifra equivalente (el único estudio con universitarios y RF no reporta F1/ROC-AUC). Esto es consistente con lo que reporta Aziz et al. (2025): menos del 35% de 46 estudios de IA en wearables publican F1-Score.

6. **Las dos fugas de datos encontradas y corregidas antes de esta sesión (`sleep_points_percentage_mean` y `_std` como features) se verificaron intactas** — ningún cambio de esta sesión las reintrodujo.

## Qué no se tocó / sigue pendiente

- La etiqueta de riesgo sigue siendo el proxy de `sleep_points_percentage` (LifeSnaps no tiene PSQI real) — limitación #1 sin cambios, sigue esperando los datos de la Capa A.
- El párrafo de discusión del Cap. 4 (`discusion_cap4_metricas.md`) está redactado pero **no incorporado al `.docx` del TI** — pendiente de tu revisión y de verificar cada cita/cifra contra el PDF original (vienen de un resumen asistido de NotebookLM, no de que yo haya leído los papers).
- La cobertura de encuestas STAI/PANAS es parcial (53/71 y 51/71 personas) — si mejora en el futuro, vale la pena reintentar esa integración.

## Estructura de esta carpeta (`poc1/`)

```
poc1/
├── README.md                     — documentación completa: metodología, resultados, limitaciones, benchmark de literatura, validación adicional
├── FINDINGS_POC1.md              — este archivo
├── discusion_cap4_metricas.md    — borrador de discusión para el TI, NO incorporado aún al .docx
├── poc_c2_random_forest.py       — pipeline oficial de C2 (carga real, feature engineering, etiqueta proxy, RF + Repeated Stratified 5-Fold, payload JSON para el LLM)
├── poc_c2_model_comparison.py    — comparación LR / RF / XGBoost, misma metodología
├── checks/                       — scripts de verificación ad hoc (NO forman parte del pipeline oficial; se guardan para que los hallazgos sean reproducibles)
│   ├── robustness_check.py       — demuestra la varianza de un solo K-Fold y el experimento STAI/PANAS descartado
│   └── permutation_check.py      — compara Gini vs. permutation importance
├── c2_output_sample.json         — payload de ejemplo (3 estudiantes) que C2 le entregaría a C3, con los números ya corregidos
└── rais_anonymized/, rais_anonymized.zip — dataset LifeSnaps completo (CSV diario/horario + encuestas + Mongo dump), datos reales, n=71
```

## Nota sobre la carpeta original

`C:\develop\tesis\POC 1\` (con espacio en el nombre) se dejó intacta como respaldo del estado previo a esta sesión. `poc1\` (sin espacio) es la copia organizada y actualizada — de aquí en adelante el trabajo continúa en esta carpeta.
