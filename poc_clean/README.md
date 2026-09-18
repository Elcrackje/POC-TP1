# poc_clean/ — versiones limpias para mostrarle al asesor

Esta carpeta tiene las versiones "de presentación" de cada PoC: mismo pipeline y mismos
resultados ya validados en `poc1/`, `poc2/`, etc., pero reorganizados en notebooks documentados,
pensados para correr por partes y mostrar avances al asesor — no para introducir nueva
metodología. Las decisiones metodológicas y el detalle técnico completo siguen viviendo en las
carpetas originales (`../poc1/`, `../poc2/`); ver [`../CLAUDE.md`](../CLAUDE.md) para el contexto
completo del proyecto.

## Contenido

- [`POC1/`](POC1/) — ML1 (Random Forest, con comparación contra Logistic Regression y XGBoost)
  sobre datos reales de LifeSnaps (n=71). Ver [`POC1/README.md`](POC1/README.md).
