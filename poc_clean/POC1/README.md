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
8. Tabla y gráfico comparativo de los tres modelos.
9. Feature importance del modelo elegido: Gini vs. Permutation importance.
10. Ejemplo del payload que ML1 le entregaría al LLM.
11. Conclusiones y limitaciones honestas.
12. Próximos pasos.

Todos los números fueron verificados contra los resultados ya oficiales de `poc1/` — son
idénticos (F1 = 0.478 ± 0.077, ROC-AUC = 0.728 ± 0.033 para Random Forest).

**Recordatorio importante que ya está en el notebook, pero vale repetirlo:** la etiqueta de riesgo
es un proxy (sleep score de Fitbit), no el PSQI real. Estos resultados no son el resultado final
para el TI — sirven para mostrar que el pipeline funciona y comparar modelos, mientras se
consiguen los datos reales de la encuesta PSQI (Capa A).

## Pendiente: ¿Deep Learning u otros modelos según la literatura?

No se agregó ninguna comparación con Deep Learning en este notebook. Con n=71 personas, un
modelo de Deep Learning muy probablemente no es apropiado (altísimo riesgo de sobreajuste con tan
pocos datos) — pero esa es una intuición, no algo que se deba meter al TI sin respaldo de tu
propia revisión de literatura.

Usa el siguiente prompt en tu herramienta de literatura (la que ya tiene cargados tus PDFs/fuentes
de la tesis) y pásame la respuesta — con eso se agrega la comparación correspondiente sin inventar
nada:

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

Cuando tengas la respuesta, la comparación adicional (si corresponde) se agrega como una sección
más del notebook, con las mismas 20 repeticiones de Repeated Stratified K-Fold que ya usan los
otros tres modelos, para que siga siendo comparable.
