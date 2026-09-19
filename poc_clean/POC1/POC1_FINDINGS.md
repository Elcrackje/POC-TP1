# POC1 — Hallazgos (versión limpia)

**Fecha:** 2026-09-19. **Dataset:** LifeSnaps (Fitbit Sense), n=71 personas reales.
**Notebook:** [`POC1_HapSleep.ipynb`](POC1_HapSleep.ipynb). Este documento complementa (no reemplaza) a
[`../../poc1/FINDINGS_POC1.md`](../../poc1/FINDINGS_POC1.md), que no se modificó.

## 1. Resumen ejecutivo

- POC1 prueba que el **pipeline de ML1** (resumen por persona, control de fugas, validación repetida, comparación
  de 4 modelos, ranking de factores para el LLM) funciona con datos reales de reloj.
- **Random Forest** queda como modelo de ML1 (F1 0.478 ± 0.077, ROC-AUC 0.728 ± 0.033), elegido por interpretabilidad.
- **Los números no son el resultado final de la tesis**, porque la etiqueta de riesgo es un proxy débil (sección 5).
  El resultado con valor científico saldrá de entrenar con el PSQI real de la Capa A.

## 2. Datos

| Aspecto | Valor |
|---|---|
| Filas / columnas | 7,410 filas persona-día, 63 columnas (se usan 12) |
| Personas | 71 |
| Período | 2021-04-08 a 2022-01-22; cada persona aporta 64–244 días (mediana 88) |
| Personas con las 11 variables predictoras | 23 de 71 (en promedio 9.1 de 11 por persona) |

Las 11 variables predictoras (todas del Fitbit Sense) y su cobertura:

| Variable | Qué mide | Filas con dato (de 7,410) | Personas |
|---|---|---|---|
| `minutesAsleep` | Minutos dormido | 3,551 (48%) | 69 |
| `minutesAwake` | Minutos despierto en la noche | 3,551 | 69 |
| `sleep_efficiency` | % del tiempo en cama dormido | 3,551 | 69 |
| `minutesToFallAsleep` | Latencia de sueño (casi siempre 0: 99.4% ceros; variable inútil) | 3,551 | 69 |
| `resting_hr` | Frecuencia cardíaca en reposo | 4,422 (60%) | 71 |
| `nremhr` | Pulso en sueño no-REM | 2,475 (33%) | 43 |
| `rmssd` | Variabilidad de la frecuencia cardíaca (HRV) | 2,475 (33%) | 43 |
| `full_sleep_breathing_rate` | Respiraciones por minuto al dormir | 2,495 (34%) | 43 |
| `spo2` | Saturación de oxígeno | 1,270 (17%) | 28 |
| `steps` | Pasos por día | 4,777 (64%) | 71 |
| `sedentary_minutes` | Minutos sedentarios (proxy de inactividad, no de pantalla) | 7,083 (96%) | 71 |

Calidad de datos:
- **Solo se convierte a número.** No se quitan outliers ni ceros. Los ceros sospechosos son pocos (1–3% en HRV,
  `nremhr`, respiración, `minutesAwake`; 86 días con 0 pasos).
- **Pocas noches en algunas personas:** mediana de 58 noches con dato de sueño, pero 9 personas tienen menos de 10.
  Su media y desviación son poco confiables y pesan igual que las de 220 noches.
- **Relleno:** tras resumir por persona, 354 de 1,704 celdas (20.8%) quedan vacías y se rellenan con la mediana
  de las 71 personas. En `spo2` es el 61% de las personas; en HRV, `nremhr` y respiración, el 39%.
  Los pesos de esas variables en el modelo no son confiables porque en buena parte son relleno.

## 3. Metodología

1. **Resumen por persona:** media y desviación estándar de cada variable → 24 columnas por persona. La desviación
   mide inconsistencia noche a noche (`minutesAsleep_std` se llama `sleep_duration_inconsistency`). Se basa en que el
   riesgo es un rasgo estable de la persona (r=0.727 en el análisis previo de autocorrelación).
2. **Etiqueta (proxy):** riesgo alto = 1 si el promedio de `sleep_points_percentage` de la persona está en el 25%
   más bajo (umbral 0.611). 18 personas de riesgo alto (25.4%).
3. **Control de fugas:** se excluyen TODAS las variantes (`_mean`, `_std`) de la columna usada para la etiqueta →
   22 features. Dos fugas reales se encontraron y corrigieron en versiones anteriores.
4. **Evaluación:** Repeated Stratified 5-Fold × 20 repeticiones (se separan personas, no días), umbral 0.5,
   métricas como media ± desviación. Un solo 5-Fold varía mucho con n=71 (el F1 osciló entre 0.29 y 0.61).
   Baseline: un modelo que siempre dice "bajo riesgo" (F1 = 0; accuracy engañosa de ~75%).
5. **Modelos:** Random Forest (300 árboles, profundidad 5), Regresión Logística (con escalado), XGBoost
   (200 árboles, profundidad 3) y SVM (kernel RBF, con escalado), todos con pesos de clase balanceados.

## 4. Resultados

| Modelo | Precisión | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.402 ± 0.070 | 0.547 ± 0.100 | 0.462 ± 0.078 | 0.669 ± 0.051 | 0.428 ± 0.059 |
| **Random Forest (elegido)** | 0.531 ± 0.079 | 0.439 ± 0.084 | **0.478 ± 0.077** | 0.728 ± 0.033 | 0.481 ± 0.046 |
| XGBoost | 0.503 ± 0.088 | 0.356 ± 0.077 | 0.414 ± 0.077 | 0.755 ± 0.037 | 0.493 ± 0.049 |
| SVM | 0.191 ± 0.188 | 0.058 ± 0.062 | 0.088 ± 0.092 | 0.668 ± 0.039 | 0.397 ± 0.044 |

- Ningún modelo gana en todo. Random Forest se elige por interpretabilidad: permite entregarle al LLM un ranking de
  factores de riesgo explicable.
- **SVM** da un F1 muy bajo con ROC-AUC razonable; la causa probable es la calibración interna de `predict_proba`
  (Platt scaling) con una clase minoritaria tan chica. Se reporta tal cual, sin ajustar el umbral.
- **Deep Learning se descartó con respaldo de literatura** (consulta a las fuentes de la tesis vía NotebookLM): ninguna
  fuente lo usa ni recomienda para clasificación de riesgo con datos tabulares agregados por persona con n<200; sus
  usos en el dominio son sobre señales crudas época por época. SVM sí lo respalda la literatura (Aziz et al. 2025:
  3er algoritmo más usado en 46 estudios de wearables, 26.1%), por eso se agregó.
- **No existe un benchmark de la literatura directamente comparable** (n≈70–100 universitarios, riesgo por persona,
  Random Forest). Detalle en `../../poc1/README.md`.

**Variables con más peso (Gini importance, promedio de 20 ajustes):**
`minutesAwake_std` (0.148), `sleep_duration_inconsistency` (0.112), `sedentary_minutes_mean` (0.100),
`steps_std` (0.060), `resting_hr_mean` (0.054). Con permutation importance el primer puesto cambia a
`sedentary_minutes_mean`, así que el "ganador" exacto es frágil. Lo estable es el grupo: inconsistencia del
sueño/vigilia noche a noche e inactividad. Al LLM se le entrega el ranking completo ("factores principales" en plural).
Las más bajas son `minutesToFallAsleep` (~0.002) y las de SpO2, HRV, `nremhr` y respiración (~0.016–0.041; ver la
advertencia sobre relleno en la sección 2).

## 5. Hallazgo principal: qué es realmente la etiqueta

`sleep_points_percentage` **no es el Sleep Score de Fitbit (1–100)**. La evidencia, toda verificada:

| Evidencia | Qué muestra |
|---|---|
| Tabla 2 del data descriptor de LifeSnaps (Yfantidou et al., 2022) | No lista ningún Sleep Score. Lista el **Stress Score** ("estima cómo responde el cuerpo al estrés a partir de frecuencia cardíaca, sueño y nivel de actividad"): 36 usuarios, 1,911 respuestas. El sueño aparece como un tipo aparte: 66 usuarios |
| Cobertura en el CSV | `sleep_points_percentage`, `exertion_points_percentage`, `responsiveness_points_percentage` y `stress_score` existen exactamente en las **mismas 1,876 filas y las mismas 37 personas** (100% de solapamiento). Encaja con los 36 usuarios del Stress Score y no con los 66 del sueño |
| Valores | Solo **19 valores distintos, en pasos de 1/30** (0.433, 0.467, 0.500, 0.533…): puntos sobre 30. Un Sleep Score de 1–100 daría valores mucho más finos |
| Diseño del Stress Management Score de Fitbit | Se compone de Responsiveness (sobre 30), Exertion Balance (sobre 40) y **Sleep Patterns (sobre 30)** ([Fitbit Community](https://community.fitbit.com/t5/The-Pulse-Fitbit-Community-Blog/Learn-More-The-Stress-Management-Score-Part-2/ba-p/5398693)). Sleep Patterns considera el sueño profundo de la noche previa, la fragmentación y una "reserva de sueño" de la semana anterior |
| Relación con variables de sueño reales (Spearman, filas con ambos datos) | `minutesAsleep` 0.31, `sleep_efficiency` 0.25, `resting_hr` 0.16, **`minutesAwake` 0.01**. Con `stress_score` 0.71. Es una relación débil con el sueño de la noche |

**Conclusión:** casi seguro es el componente *Sleep Patterns* (puntos sobre 30) del Stress Management Score. El
Sleep Score de 1–100 que describe la ayuda de Google (duración, fases y restauración de una noche) es otra métrica y
no está en LifeSnaps. La equivalencia exacta sigue sin confirmarse en un diccionario oficial: ni el paper ni el
repositorio de LifeSnaps describen esa columna.

**Consecuencias para poc1:**
1. **Cobertura:** solo 37 de 71 personas tienen dato. Las otras 34 reciben la mediana (0.62), que queda sobre el umbral
   (0.611): quedan etiquetadas "bajo riesgo" sin medición. Las 18 personas de riesgo alto son todas de las 37 con
   dato. Entre esas 37 la prevalencia real de "riesgo alto" es **48.6%**, no 25%.
2. **No mide la calidad de sueño de una noche.** Mezcla sueño con otras señales y usa una reserva semanal.
3. **Posible circularidad:** el Stress Score se calcula con frecuencia cardíaca, sueño y actividad, y las predictoras
   incluyen esas tres cosas. Habría superposición conceptual, más allá de la fuga técnica ya corregida. Esto obliga a
   declarar como limitación que `minutesAwake_std` y las demás no se pueden citar como "factores de mala calidad de
   sueño" sin matizar.
4. Si la columna fuera un Sleep Score, la superposición también existiría (ese score usa duración e inquietud, que son
   predictoras). Es decir, el problema de fondo no depende de cuál de las dos lecturas sea la correcta.

## 6. Qué demuestra y qué no

**Sí demuestra:**
- El pipeline completo funciona sobre datos reales de reloj, y solo cambia la función de la etiqueta cuando llegue el PSQI.
- La metodología de evaluación es necesaria con n chico (varianza de un solo K-Fold).
- La comparación de 4 modelos es justa, con respaldo de literatura para descartar Deep Learning.
- Qué exigir en el estudio real: SpO2 falta en el 83% de las filas, y hay personas con pocas noches.

**No demuestra:**
- Que se predice mala calidad de sueño real.
- Que el ROC-AUC de 0.73 sea sólido (ver riesgos).

**Riesgos y sospechas sin comprobar:**
1. El ROC-AUC podría inflarse si el modelo detecta "personas con muchos datos faltantes": las 34 sin etiqueta real
   tienen además features rellenadas con la mediana y coinciden con la etiqueta 0.
2. La circularidad descrita arriba.
3. **Fuga leve de preprocesamiento:** la mediana de relleno se calcula con las 71 personas antes de dividir en
   entrenamiento y prueba. No se midió su efecto.

## 7. Siguientes pasos propuestos

**A. Verificaciones rápidas de poc1 (sin datos nuevos, una vez que lo apruebes):**
1. Reevaluar con **solo las 37 personas con etiqueta real** y comparar contra 0.73.
2. **Prueba de "detector de faltantes":** entrenar el mismo modelo para predecir "¿esta persona tiene etiqueta real?"
   con las mismas features. Si acierta mucho, el ROC-AUC de poc1 está inflado por faltantes.
3. **Prueba de circularidad:** repetir sin las features de frecuencia cardíaca y actividad (`resting_hr`, `steps`,
   `sedentary_minutes`) y ver cuánto cae el rendimiento.
4. Mover el relleno con la mediana **dentro** de cada partición (elimina la fuga leve).

**B. Documentación (antes de mostrar el notebook al asesor):**
5. Corregir las celdas del notebook que aún llaman a la columna "puntaje de sueño de Fitbit", y reflejar las
   limitaciones de la sección 5 en `poc1/README.md`, `poc1/FINDINGS_POC1.md` y `CLAUDE.md`.

**C. Lo que sostiene la tesis (empezar ya: tiene el plazo más largo):**
6. **Plan de la Capa A (PSQI):** n≥50; PSQI (mala calidad si ≥5) y datos de reloj del **mismo mes** que evalúa el
   cuestionario; definir un mínimo de noches por persona (poc1 mostró que menos de 10 es ruidoso; acordar el umbral con
   el asesor); definir el conjunto de variables y el dispositivo o dispositivos. Considerar ética y consentimiento.
7. Fijar antes de recolectar el protocolo de análisis (validación repetida, métricas, control de fugas), para no
   ajustarlo después de ver resultados.

**D. Otros PoC y pendientes:**
8. **PoC multi-wearable:** ya investigado en [`../multiwearable/variables_por_plataforma.md`](../multiwearable/variables_por_plataforma.md).
9. **PoC del LLM (poc3):** recomendaciones a partir del payload de ML1.
10. Verificar `../../poc1/discusion_cap4_metricas.md` contra las fuentes primarias.

**Orden sugerido:** A (1–4) → B (5) → arrancar C en paralelo → poc3 (LLM) → multi-wearable.

## 8. Reproducibilidad

- Entorno: Python 3.11.9, venv en `.venv/` (ver `CLAUDE.md`). El notebook corre de arriba hacia abajo y ya guarda sus salidas.
- Semilla 42; los números del notebook coinciden con los del pipeline oficial en `poc1/`.
- Los datos de LifeSnaps son públicos ([Zenodo](https://zenodo.org/records/7229547)); la copia usada está en `data/`.
