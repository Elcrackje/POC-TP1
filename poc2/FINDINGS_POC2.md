# Findings — PoC 2 (ML1 con datos sintéticos de wearable)

Resumen para compartir de qué se hizo en este PoC y qué se encontró. Contexto rápido: HapSleep tiene un modelo (ML1, Random Forest) que clasifica el riesgo de mala calidad de sueño de una persona a partir de datos de wearable, y un LLM que genera recomendaciones a partir de ese output. Ya existe un PoC 1 con datos reales (dataset LifeSnaps, n=71). Este PoC 2 usa datos **100% sintéticos** — generados por nosotros, no de ningún dataset público — para poder validar algo que con datos reales es imposible de comprobar.

## Por qué datos sintéticos, si ya hay un PoC con datos reales

Con datos reales nunca sabes con certeza si el modelo está capturando la señal correcta o solo ruido que casualmente correlaciona con la etiqueta que usas para entrenar. Al generar los datos nosotros mismos, sí sabemos la verdad — le pusimos un "riesgo real" (`latent_risk`) a cada persona sintética al construir el dataset, y ese número **nunca se le pasa al modelo**. Después de entrenar, comparamos qué tan bien la predicción del modelo se parece a esa verdad. Eso es una validación de que el pipeline (cómo se procesan los datos, cómo se arma el modelo, cómo se evalúa) está bien construido — no es una medición de qué tan bien va a funcionar en producción.

## Qué se hizo

1. Se generaron 150 personas sintéticas (más que las 71 reales de LifeSnaps), cada una con un riesgo de sueño fijo (`latent_risk`, entre 0 y 1, sesgado hacia bajo riesgo con cola de alto riesgo) — esto replica el hallazgo ya validado en la tesis de que el riesgo es un rasgo estable de la persona, no algo que cambia noche a noche.
2. Para cada persona se generaron entre 30 y 120 días de datos con las mismas columnas que un wearable real (HRV, frecuencia cardiaca, SpO2, métricas de sueño, pasos, sedentarismo), correlacionadas con su `latent_risk` más ruido.
3. Detalle clave del diseño: no solo el promedio de minutos despierto/sedentarismo sube con el riesgo — también se vuelven **más inconsistentes noche a noche** a mayor riesgo. Esto se hizo a propósito porque en el PoC 1 (datos reales) ese fue justamente el hallazgo: la inconsistencia noche a noche fue el factor más predictivo.
4. Se corrió el mismo pipeline de ML1 (Random Forest) que en el PoC 1, con la misma metodología: Repeated Stratified 5-Fold Cross-Validation (20 repeticiones, no un solo split) — porque con muestras chicas un solo split puede dar resultados muy distintos solo por azar.
5. Se comparó la predicción del modelo contra el `latent_risk` real (el "ground truth" que solo existe porque los datos son sintéticos).

## Resultados

**Desempeño del modelo (Repeated Stratified 5-Fold, 20 repeticiones, n=150):**

| Métrica | Random Forest (mean ± std) | Baseline trivial |
|---|---|---|
| Precisión | 0.837 ± 0.021 | 0.000 |
| Recall | 0.904 ± 0.027 | 0.000 |
| **F1** | **0.869 ± 0.019** | 0.000 |
| ROC-AUC | 0.988 ± 0.002 | — |
| PR-AUC | 0.967 ± 0.005 | — |

**Validación contra el riesgo real (solo posible con datos sintéticos):**

| Chequeo | Resultado |
|---|---|
| Correlación (predicción del modelo vs. riesgo real inyectado) | **0.898** |
| ROC-AUC del modelo contra el riesgo real (no la etiqueta proxy) | **0.991** |
| Qué tan de acuerdo están la etiqueta proxy y el riesgo real | 97.3% |

**Factores más importantes que encontró el modelo:** eficiencia de sueño promedio, minutos despierto (promedio y variabilidad), sedentarismo promedio, y latencia para dormirse — coincide con las variables que se usaron para construir el riesgo al generar los datos. Esto confirma que el modelo está aprendiendo la señal correcta, no artefactos raros del dataset.

## Insights (lo importante para llevarse)

1. **El pipeline funciona como se espera.** Cuando existe una señal de riesgo real en los datos, el modelo la recupera casi perfectamente (correlación 0.898, ROC-AUC 0.991 contra la verdad). Esto da confianza en que la arquitectura (cómo se agregan los datos por persona, qué features se calculan, cómo se entrena y evalúa el Random Forest) está bien diseñada.

2. **Ojo — estos números NO son comparables con el PoC 1 de datos reales.** El F1=0.869 de acá es mucho más alto que el F1=0.478 del PoC 1 con datos reales, y es exactamente lo esperado: los datos sintéticos tienen una relación limpia (con ruido controlado) entre las variables y el riesgo, mientras que la vida real tiene datos faltantes, medición imperfecta, y factores que no estamos capturando. **No hay que citar el 0.869 como "cuánto va a funcionar el modelo" — el número que importa para eso es el del PoC 1.**

3. **Hallazgo lateral importante:** incluso en este mundo sintético "limpio", la etiqueta que se usa como proxy de calidad de sueño solo coincide 97.3% con el riesgo real — hay un 2.7% de desacuerdo aunque el mundo es controlado por nosotros. En el PoC 1, donde ese proxy es la única etiqueta disponible porque no hay una encuesta real de calidad de sueño (PSQI) todavía, ese desacuerdo en la vida real casi seguro es mayor a 2.7%. Esto refuerza por qué conseguir los datos reales de la encuesta (Capa A del proyecto) es importante — el proxy que se usa hoy tiene un límite de qué tan bien puede representar el riesgo real, incluso en el mejor de los casos.

## Archivos de este PoC

- `generate_synthetic_wearable_data.py` — genera los datos sintéticos y el ground truth.
- `poc_ml1_random_forest.py` — corre el modelo y hace la validación contra el ground truth.
- `synthetic_wearable_data.csv` / `synthetic_ground_truth.csv` — datos generados (reproducibles corriendo el primer script).
- `README.md` — documentación técnica completa de este PoC.

## Qué sigue

- Este PoC fue solo del modelo de clasificación de riesgo (ML1). El LLM que genera recomendaciones a partir del output de ML1 va en un PoC aparte.
- Si se quiere estresar más el pipeline (no solo confirmarlo), se puede repetir este PoC con más ruido, datos faltantes inyectados, o una relación no lineal entre features y riesgo, para ver en qué punto deja de funcionar.
