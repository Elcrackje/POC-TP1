# poc_clean/ — versiones limpias para mostrarle al asesor

Esta carpeta tiene las versiones "de presentación" de cada PoC: mismo pipeline y mismos
resultados ya validados en `poc1/`, `poc2/`, etc., pero reorganizados en notebooks documentados,
pensados para correr por partes y mostrar avances al asesor — no para introducir nueva
metodología. Las decisiones metodológicas y el detalle técnico completo siguen viviendo en las
carpetas originales (`../poc1/`, `../poc2/`); ver [`../CLAUDE.md`](../CLAUDE.md) para el contexto
completo del proyecto.

## Contenido

- [`POC1/`](POC1/) — ML1 (Random Forest, con comparación contra Logistic Regression, XGBoost y SVM)
  sobre datos reales de LifeSnaps (n=71). Ver [`POC1/README.md`](POC1/README.md).
- [`POC1_1/`](POC1_1/) — POC 1.1 (validación solo con las 37 personas que tienen etiqueta real) y POC 1.2 (quitar
  grupos de variables). Notebook: [`POC1_1/POC1_1_Validacion.ipynb`](POC1_1/POC1_1_Validacion.ipynb). Hallazgos en
  [`POC1/POC1_FINDINGS.md`](POC1/POC1_FINDINGS.md).
- [`POC1_3/`](POC1_3/) — POC 1.3: en vez de una etiqueta por percentil, el modelo predice un puntaje continuo y el
  riesgo se obtiene con un corte fijo. Incluye la explicación de cómo se calcula el valor y un ejemplo de la salida de
  ML1. Notebook: [`POC1_3/POC1_3_Puntaje_continuo.ipynb`](POC1_3/POC1_3_Puntaje_continuo.ipynb).
- [`multiwearable/`](multiwearable/) — investigación (sin PoC corrido) de qué variables entrega cada marca de reloj
  Android (Fitbit, Samsung, Amazfit, Xiaomi) vía Health Connect / Google Health API, para un futuro PoC
  multi-wearable. Ver [`multiwearable/variables_por_plataforma.md`](multiwearable/variables_por_plataforma.md).
