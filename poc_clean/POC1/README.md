# POC1 (versión limpia) — ML1: riesgo de mala calidad de sueño

Versión ordenada y documentada de [`../../poc1/`](../../poc1/) para mostrarle al asesor. **No
cambia ninguna decisión metodológica** respecto a la versión original — mismos hiperparámetros,
misma metodología de evaluación, mismos resultados. Solo está reorganizada en un notebook que se
puede correr y leer por partes, con cada paso documentado en su propia celda.

## Qué hay en esta carpeta

- **`POC1_HapSleep.ipynb`** — el notebook. Ábrelo con Jupyter (`jupyter lab` o `jupyter notebook`
  desde esta carpeta, con el venv activado) y corre `Cell → Run All`, o celda por celda si quieres
  inspeccionar resultados intermedios. Ya tiene los resultados y gráficos guardados adentro, así
  que también se puede abrir y leer sin volver a correrlo.
- **`data/daily_fitbit_sema_df_unprocessed.csv`** — copia del dataset real (LifeSnaps, n=71) que
  usa el notebook. Es el mismo archivo que está en `poc1/rais_anonymized/...` — se copió aquí para
  que esta carpeta sea autocontenida y no dependa de rutas relativas frágiles.

## Qué contiene el notebook

1. Carga de datos + limitaciones del dataset.
2. Feature engineering (agregación a nivel de persona).
3. Construcción de la etiqueta de riesgo (proxy) + chequeo de fuga de datos.
4. Metodología de evaluación (Repeated Stratified 5-Fold × 20 repeticiones) y por qué.
5. **Modelo 1 — Random Forest** (el elegido para producción): resultados.
6. **Modelo 2 — Regresión Logística**: resultados.
7. **Modelo 3 — XGBoost**: resultados.
8. **Modelo 4 — SVM**: resultados (agregado con respaldo de literatura, ver abajo).
9. Tabla y gráfico comparativo de los cuatro modelos.
10. Feature importance del modelo elegido: Gini vs. Permutation importance.
11. Ejemplo del payload que ML1 le entregaría al LLM.
12. Conclusiones y limitaciones honestas.
13. Próximos pasos.

Todos los números fueron verificados contra los resultados ya oficiales de `poc1/` — son
idénticos (F1 = 0.478 ± 0.077, ROC-AUC = 0.728 ± 0.033 para Random Forest).

**Recordatorio importante que ya está en el notebook, pero vale repetirlo:** la etiqueta de riesgo
es un proxy (sleep score de Fitbit), no el PSQI real. Estos resultados no son el resultado final
para el TI — sirven para mostrar que el pipeline funciona y comparar modelos, mientras se
consiguen los datos reales de la encuesta PSQI (Capa A).

## Deep Learning vs. SVM — ya resuelto con literatura (no es una suposición)

Se le preguntó directamente a la herramienta de literatura de la tesis (NotebookLM, con las
fuentes cargadas del proyecto) el siguiente prompt:

```
Contexto: mi tesis predice riesgo de mala calidad de sueño en estudiantes universitarios a
partir de datos agregados por persona de un smartwatch (frecuencia cardíaca, HRV, SpO2,
duración/eficiencia de sueño, pasos, sedentarismo — media y desviación estándar por persona
a lo largo de varios días). El dataset tiene n=71-150 personas (tamaño de muestra chico). Ya
comparé Random Forest, Regresión Logística y XGBoost con Repeated Stratified 5-Fold CV (20
repeticiones), reportando F1, ROC-AUC y PR-AUC contra un baseline trivial.

Preguntas, basándote SOLO en las fuentes que tengo cargadas (cita autor/año para cada
afirmación, no inventes ni generalices de memoria):

1. ¿Alguna de mis fuentes usa o recomienda Deep Learning (redes neuronales, LSTM, etc.) para
   clasificación de riesgo/calidad de sueño con un tamaño de muestra similar (n<200 personas,
   agregado por persona, no por época/ventana de tiempo)? Si sí, ¿qué arquitectura y con qué
   resultado comparado a modelos clásicos como Random Forest?
2. ¿Alguna de mis fuentes advierte explícitamente sobre el riesgo de sobreajuste de Deep
   Learning con muestras chicas en este dominio (wearables + sueño)?
3. ¿Hay algún otro modelo de ML (no necesariamente Deep Learning) que mis fuentes sugieran
   como más apropiado para este tamaño de muestra y este tipo de features agregadas, que yo
   no haya probado todavía?
4. Si ninguna de mis fuentes toca este tema directamente, dilo explícitamente en vez de
   generalizar con conocimiento general de machine learning.
```

**Respuesta (resumen; el detalle completo con citas está en el chat de la tesis, no reproducido
aquí para no duplicar el texto largo):**

1. **Ninguna fuente usa o recomienda Deep Learning** para clasificar riesgo sobre datos
   tabulares agregados por persona con n<200. Las fuentes que sí usan DL (ej. Kim et al. 2025,
   "SleepWatcher") lo hacen sobre señales crudas época-por-época (segundos/minutos) — otra
   tarea distinta a la de este PoC. Para datos tabulares/agregados, la literatura recurre a
   Random Forest (30.4% de 46 estudios, Aziz et al. 2025) o SVM (26.1%).
2. No hay una advertencia textual explícita sobre sobreajuste de DL en este dominio con n
   chica, pero sí hay consenso indirecto: Aziz et al. (2025) recomienda K-Fold CV para muestras
   chicas, y de Zambotti et al. (2024) / Chee et al. (2025) advierten sobre falta de
   generalización de modelos entrenados en muestras pequeñas o de conveniencia.
3. **SVM** es el modelo clásico que falta probar — 3er algoritmo más usado en la revisión de
   Aziz et al. (2025), justo detrás de RF. Ya se agregó a la comparación (ver notebook).
4. Confirmado explícitamente: ninguna fuente evalúa DL para este caso de uso específico. El
   diseño experimental actual (Repeated Stratified 5-Fold × 20, F1/ROC-AUC/PR-AUC vs. baseline)
   es, según la misma respuesta, "el esquema metodológico más riguroso y alineado con las
   recomendaciones" de las fuentes revisadas.

**Conclusión aplicada:** no se agregó Deep Learning (sin respaldo de literatura para este n y
este tipo de dato). Se agregó **SVM** a la comparación de modelos, con el mismo protocolo de
evaluación (Repeated Stratified 5-Fold × 20). Resultado: F1 = 0.088 ± 0.092, ROC-AUC = 0.668 ±
0.039, PR-AUC = 0.397 ± 0.044 — el ROC-AUC es razonable (similar al de LR), pero el F1 sale muy
bajo, probablemente por inestabilidad en la calibración interna de `predict_proba` de `SVC`
(Platt scaling) con una clase minoritaria tan chica (~18 personas en n=71). Se reporta tal cual,
sin ajustar el umbral de decisión para mejorar el número — el objetivo es una comparación justa
entre los cuatro modelos, no maximizar el resultado de uno en particular.

La misma respuesta de NotebookLM mencionó, como sugerencia adicional, correr también SVM sobre
el mismo protocolo — ya se hizo, es el modelo 4 del notebook.
