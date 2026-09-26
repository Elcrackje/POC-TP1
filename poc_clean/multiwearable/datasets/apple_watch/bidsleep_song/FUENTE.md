# A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels (BIDSleep)

- **Enlace:** https://physionet.org/content/bidsleep-dataset/1.0.0/
- **Tipo:** 3 (Apple Watch contra diadema EEG Dreem 2)
- **Marca y modelo:** Apple Watch (modelo no especificado) y diadema Dreem 2 como referencia.
- **Personas / noches:** 47 voluntarios adultos sanos; 3 a 7 noches por persona; 253 noches en total.
- **Variables:** frecuencia cardíaca instantánea (PPG, ~0.2 Hz) y acelerometría de 3 ejes. **No hay pasos** ni resúmenes diarios de sueño.
- **Etiqueta:** etapas de sueño (Wake, N1, N2, N3, REM, Unknown) en épocas de 30 s, de la diadema Dreem 2; un experto revisó los hipnogramas.
  NO es calidad de sueño percibida ni PSQI.
- **Población:** adultos sanos sin antecedentes de trastornos del sueño ni cardiovasculares.
- **Licencia y descarga:** Open Data Commons Attribution License v1.0. La página dice que cualquiera puede acceder cumpliendo la licencia;
  no menciona registro obligatorio (verificar al descargar). Opciones wget y AWS CLI.
- **Tamaño:** **27.9 GB descomprimido.** Por carpeta de sujeto: `motion.csv`, `hr.csv`, `labels.mat` por noche.
- **Publicado:** 12 mayo 2026 (v1.0.0). Autor: Tzu-An Song.
- **Paper:** Song et al. (2025), "AI-driven sleep staging using instantaneous heart rate and accelerometry: Insights from an Apple Watch study", IEEE TBME. https://doi.org/10.1109/TBME.2025.3612158
- **Fecha de descarga:**

## Ojo
- Sirve para estudiar cuán bien el Apple Watch estima etapas de sueño contra EEG, no para entrenar riesgo de mala calidad de sueño.
- Para sacar duración o eficiencia habría que derivarlas de las señales crudas: es otra tarea (clasificación de sueño), fuera del alcance actual.
- Apple queda para una fase posterior. Prioridad baja; descargar solo si hay espacio (28 GB) y una pregunta concreta.
- Correcciones a lo que dijo el chat: la página indica 2026 con paper ya publicado en 2025 (no "en revisión"), y no exige registro explícito.
