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
- **POC 1.1 (sección 6b):** usando solo las 37 personas con etiqueta real, el ROC-AUC de Random Forest sube de 0.728 a
  0.936 (0.830 con una variante de umbral). Las 34 personas rellenadas rebajaban el resultado de POC1: el 0.73 es
  una cota conservadora, no un techo.

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

**Decisión de trabajo:** en los notebooks y documentos se sigue llamando a la etiqueta "puntaje de sueño de Fitbit",
porque así viene en el CSV. Esta sección solo documenta de dónde sale realmente ese valor.

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

**Riesgos y sospechas:**
1. Se sospechó que el ROC-AUC estuviera inflado por "personas con muchos datos faltantes". POC 1.1 apunta en sentido
   contrario: quitar las 34 personas rellenadas *sube* el resultado, o sea que le metían ruido. La prueba directa de
   "detector de faltantes" no se corrió (decisión del equipo).
2. La circularidad descrita arriba sigue siendo un riesgo, y POC 1.2 mostró que el modelo se apoya más en pulso y
   actividad que en las variables de sueño (sección 6b).
3. **Fuga leve de preprocesamiento:** la mediana de relleno se calcula con las 71 personas antes de dividir en
   entrenamiento y prueba. No se midió su efecto.
4. Con 37 personas los resultados de 1.1 y 1.2 son ruidosos y probablemente optimistas; además esas 37 (las que tienen
   Stress Score) podrían no representar a las otras 34.

## 6b. POC 1.1 y POC 1.2 — resultados

Notebook: [`../POC1_1/POC1_1_Validacion.ipynb`](../POC1_1/POC1_1_Validacion.ipynb). Mismo pipeline y protocolo que POC1
(Repeated Stratified 5-Fold × 20, umbral 0.5, semilla 42). El Random Forest de referencia con 71 personas reproduce
exactamente el resultado oficial.

**POC 1.1 — solo las 37 personas con etiqueta real (Random Forest).** En 1.1a se usa el umbral de POC1 (0.611); en 1.1b,
el cuartil propio de las 37 (0.512).

| Configuración | n | Prevalencia | F1 | ROC-AUC | PR-AUC (azar) |
|---|---|---|---|---|---|
| POC1 (71 personas) | 71 | 25.4% | 0.478 ± 0.077 | 0.728 ± 0.033 | 0.481 ± 0.046 (0.254) |
| 1.1a (37, umbral POC1) | 37 | 48.6% | 0.849 ± 0.036 | 0.936 ± 0.020 | 0.941 ± 0.021 (0.486) |
| 1.1b (37, cuartil propio) | 37 | 27.0% | 0.583 ± 0.067 | 0.830 ± 0.033 | 0.635 ± 0.056 (0.270) |

En 1.1a, los cuatro modelos (ROC-AUC): Logistic Regression 0.951, Random Forest 0.936, SVM 0.925, XGBoost 0.909. Sus
rangos se solapan, así que no se pueden distinguir con n=37. SVM, que fallaba en POC1 (F1 0.088), llega aquí a F1 0.862
con clases casi balanceadas.

**POC 1.2 — quitar grupos de variables (Random Forest, ROC-AUC).**

| Configuración | 71 personas (POC1) | 37 personas (1.1a) |
|---|---|---|
| Todas (22 features) | 0.728 ± 0.033 | 0.936 ± 0.020 |
| Sin pulso ni actividad (16) | 0.682 ± 0.027 | 0.820 ± 0.037 |
| Sin variables de sueño (14) | 0.664 ± 0.047 | 0.918 ± 0.023 |

**Lectura:**
- Las 34 personas rellenadas no inflaban POC1; lo rebajaban. Con etiqueta real el modelo reproduce el puntaje de sueño
  de Fitbit con un ROC-AUC de 0.83–0.94.
- Eso **no** significa que prediga calidad de sueño real: predice un puntaje de Fitbit desde otras señales del mismo reloj.
- Sin las variables de sueño el resultado casi no cambia (0.936 → 0.918); sin pulso ni actividad baja más (→ 0.820). El
  modelo se apoya más en pulso y actividad. Como las variables están correlacionadas, esta prueba no aísla el aporte
  individual de cada una.
- Ninguna quita lleva al azar: hay señal en todos los grupos.

## 6c. POC 1.3 — puntaje continuo en vez de percentil

Notebook: [`../POC1_3/POC1_3_Puntaje_continuo.ipynb`](../POC1_3/POC1_3_Puntaje_continuo.ipynb). Motivo: la etiqueta por
percentil es relativa al grupo (siempre marca ~25%), así que no funcionaría en una población donde casi todos duermen mal.
Aquí el modelo predice un **índice continuo** (1 − puntaje de sueño de Fitbit; más alto = peor, igual que el PSQI) y el
riesgo se obtiene **después** con un corte fijo. 37 personas con puntaje real, 5-Fold × 20.

| Modelo | Spearman | MAE | R² |
|---|---|---|---|
| **Random Forest** | 0.697 ± 0.050 | 0.108 ± 0.005 | 0.449 ± 0.054 |
| Ridge (lineal) | 0.709 ± 0.038 | 0.131 ± 0.010 | 0.227 ± 0.127 |
| Baseline: siempre el promedio | — | 0.165 ± 0.004 | −0.069 ± 0.047 |

- **Un solo modelo sirve para cualquier corte:** con cortes que dejan entre 22% y 78% de personas en riesgo alto, el
  ROC-AUC va de 0.79 a 0.93 (PR-AUC muy por encima del azar en todos).
- **Salida de ML1 ensayada:** índice estimado, rango de 90% (a partir de los errores pasados), probabilidad de superar el
  corte, indicador de "zona gris", factores principales y calidad de datos (noches con dato, variables sin datos). Ver el
  notebook.
- **Límites:** el rango es ancho (errores de −0.20 a +0.30 con una desviación del índice de 0.198) y es el mismo para todas
  las personas, aunque alguna tenga solo 6 noches. El ranking de variables es global e inestable con n=37 (aquí domina
  `sedentary_minutes_mean`, distinto de POC1); no citarlo como hallazgo.
- Las columnas de etapas de sueño del CSV (`sleep_*_ratio`, 44% de las filas, 61 personas) promedian ≈1 las cuatro: parecen
  razones relativas y no proporciones de la noche. Su definición hay que confirmarla antes de usarlas.

**Matiz posterior (POC 1.5, sección 6d):** la conclusión "un solo modelo sirve para cualquier corte" se comprobó con la
señal del proxy real. En el laboratorio sintético, cuando el reloj refleja poco del puntaje, aplicar el corte a un puntaje
estimado se contrae hacia el promedio y falla en poblaciones extremas.

## 6d. POC 1.5 — laboratorio sintético con verdad conocida

Notebook: [`../POC1_5/POC1_5_Sintetico.ipynb`](../POC1_5/POC1_5_Sintetico.ipynb). Datos 100% simulados (cada persona con un
puntaje verdadero 0–21, más alto = peor, corte fijo 5), estudios de 50 personas repetidos en 30 cohortes independientes.
**No mide desempeño real:** depende de los supuestos del simulador.

**Parte 1 — qué se rompe cuando cambia la población** (nivel L4: el reloj refleja solo parte del puntaje, r = 0.7):

| Población (% real en riesgo alto) | Percentil: marca / recall | Corte fijo: marca / recall | Continuo + corte: marca / recall |
|---|---|---|---|
| Duermen mal (77%) | 30% / 0.37 | 78% / 0.85 | 83% / 0.90 |
| Mixta (49%) | 30% / 0.45 | 49% / 0.68 | 40% / 0.59 |
| Duermen bien (26%) | 32% / 0.62 | 24% / 0.50 | 7% / 0.17 |

- El **percentil marca ~30% en cualquier población**; donde casi todos duermen mal deja pasar a la mayoría. Su ROC-AUC no
  cae (ordena bien): falla la decisión "riesgo alto sí o no".
- El **corte fijo** sigue a la realidad en las tres poblaciones. El **puntaje continuo** con corte se contrae hacia el promedio
  cuando la señal es débil: sobre-marca donde duermen mal y casi no detecta donde duermen bien.
- **El F1 solo engaña:** en L4 el F1 del corte fijo (0.85, 0.69, 0.51) es casi el de un modelo tonto que siempre dice "alto"
  (0.87, 0.66, 0.41). El ROC-AUC (≈ 0.75 en L4 para los tres enfoques) es la medida útil.

**Parte 2 — qué esperar con 50 personas** (enfoque continuo, población mixta, ROC-AUC promedio [rango 5–95% entre estudios]):

| Nivel | ROC-AUC |
|---|---|
| L0 limpio | 1.00 |
| L1–L3 (ruido entre personas y datos faltantes, como LifeSnaps) | 0.97 → 0.96 |
| L4 (el reloj refleja parte del puntaje, r = 0.7) | 0.75 [0.63 – 0.83] |
| L5 (r = 0.5) | 0.63 [0.50 – 0.73] |

Lo que más pesa es cuánto del puntaje refleja el reloj (valor real desconocido), no la suciedad de los datos. Con 50
personas el resultado varía unos ±0.1 solo por qué personas te toquen. Positivos esperados entre 50: ~38 si la población
duerme mal, ~24 si es mixta, ~12 si duerme bien.

**Parte 3 — tamaño** (L4): ROC-AUC 0.70 [0.48 – 0.88] con 30, 0.75 [0.63 – 0.83] con 50, 0.76 [0.72 – 0.82] con 100. Más
personas reduce sobre todo la incertidumbre, no mejora tanto el promedio.

**Implicación de diseño:** el nivel de riesgo debe salir de un corte absoluto (no percentil); el clasificador con corte fijo es
el más estable para "riesgo alto sí o no", y el puntaje continuo con su rango sirve para graduar la salida.

## 7. Siguientes pasos propuestos

**A. Verificaciones de poc1:**
1. **HECHO — POC 1.1:** reevaluación solo con las 37 personas con etiqueta real (sección 6b).
2. **DESCARTADO (decisión del equipo):** prueba de "detector de faltantes".
3. **HECHO — POC 1.2:** quitar grupos de variables. Se corrieron dos versiones: sin pulso ni actividad, y sin variables
   de sueño (la segunda se agregó porque la etiqueta es un componente de sueño).
4. **Pendiente, opcional:** mover el relleno con la mediana **dentro** de cada partición (elimina la fuga leve). Consiste
   en calcular la mediana solo con las personas de entrenamiento y aplicarla a las de prueba.

**B. Documentación:**
5. **Decisión:** se mantiene el nombre "puntaje de sueño de Fitbit" y no se editan las celdas del notebook de POC1.
   Pendiente, a decisión del equipo: reflejar la sección 5 y los resultados de 6b en `poc1/README.md`,
   `poc1/FINDINGS_POC1.md` y `CLAUDE.md`.

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
