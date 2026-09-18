# HapSleep — PoCs del motor de ML + LLM

Tesis: app que predice riesgo de mala calidad de sueño en estudiantes universitarios de Lima usando datos de smartwatch, con un LLM que genera recomendaciones a partir de ese riesgo. Arquitectura: **ML1** (clasificación de riesgo) → **LLM** (recomendaciones).

> Si estás retomando esto en otra máquina / otra sesión de Claude Code: abre esta carpeta y lee primero **[CLAUDE.md](CLAUDE.md)** — tiene el contexto completo (decisiones cerradas, fugas de datos ya corregidas, disciplina metodológica, y resumen de hallazgos de ambos PoCs) para que no se repita trabajo ya hecho.

## Carpetas

### [`poc_clean/`](poc_clean/) — versiones limpias para mostrarle al asesor

Notebooks documentados, pensados para correr por partes, con los mismos resultados ya validados
de `poc1/`/`poc2/` (no introducen metodología nueva). Empieza aquí si necesitas presentar avances.

### [`poc1/`](poc1/) — ML1 (Random Forest) sobre datos REALES

Dataset real: LifeSnaps (Fitbit Sense), n=71 estudiantes/personas reales. Resultado oficial (Repeated Stratified 5-Fold CV, 20 repeticiones):

| F1 | ROC-AUC | PR-AUC |
|---|---|---|
| 0.478 ± 0.077 | 0.728 ± 0.033 | 0.481 ± 0.046 |

Incluye: pipeline oficial, comparación LR/RF/XGBoost, permutation importance vs. Gini, contexto de benchmark en la literatura, y un borrador de discusión para el Cap. 4 del TI (pendiente de revisión, no incorporado al `.docx`).

Lectura recomendada: [`poc1/FINDINGS_POC1.md`](poc1/FINDINGS_POC1.md) (resumen de hallazgos) → [`poc1/README.md`](poc1/README.md) (documentación técnica completa).

### [`poc2/`](poc2/) — ML1 sobre datos SINTÉTICOS de wearable

Datos generados con un riesgo real (`latent_risk`) conocido por persona, para validar que el pipeline recupera la señal correcta cuando existe — algo que con datos reales no se puede comprobar. Resultado (mismo modelo, misma metodología):

| F1 | ROC-AUC | Correlación vs. riesgo real |
|---|---|---|
| 0.869 ± 0.019 | 0.988 ± 0.002 | 0.898 |

**Importante:** estos números no son comparables con los de poc1 ni deben citarse como desempeño esperado — son de validación de arquitectura, no de producción.

Lectura recomendada: [`poc2/FINDINGS_POC2.md`](poc2/FINDINGS_POC2.md) → [`poc2/README.md`](poc2/README.md).

### `POC 1/` (con espacio en el nombre)

Respaldo intacto del estado de poc1 antes de la sesión de reorganización/limpieza. No se usa activamente — el trabajo vigente está en `poc1/`.

## Pendiente

1. PoC del LLM (recomendaciones a partir del output de ML1) — no empezado todavía.
2. Verificar `poc1/discusion_cap4_metricas.md` contra fuentes primarias antes de incorporarlo al TI.
3. Reemplazar la etiqueta proxy de poc1 por el PSQI real en cuanto existan los datos de la Capa A, y volver a correr todo.

Detalle completo de cada pendiente en [CLAUDE.md](CLAUDE.md).
