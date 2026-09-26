# Motion and heart rate from a wrist-worn wearable and labeled sleep from polysomnography (Walch)

- **Enlace (fuente original, verificada):** https://physionet.org/content/sleep-accel/1.0.0/
- **Enlace alterno (Kaggle, NO verificado: la página no cargó):** https://www.kaggle.com/datasets/msarmi9/walch-apple-watch-sleep-dataset
- **Tipo:** 3 (reloj contra polisomnografía)
- **Marca y modelo:** Apple Watch
- **Personas:** 31 (39 reclutadas; 8 excluidas por errores de transmisión, apnea o trastorno de conducta REM).
- **Variables:** aceleración cruda (x, y, z en g), frecuencia cardíaca (bpm), pasos. Etiquetas: etapas de sueño por polisomnografía (Wake, N1, N2, N3, REM).
- **Etiqueta:** polisomnografía. NO es calidad de sueño percibida ni PSQI.
- **Licencia y descarga:** Open Data Commons Attribution License v1.0. Sin registro: ZIP directo (550.1 MB; 2.2 GB descomprimido), o `wget`/AWS S3 según la página.
- **Publicado:** 8 octubre 2019 (v1.0.0). Autora: Olivia Walch.
- **Paper:** Walch, Huang, Forger y Goldstein (2019), "Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device", *Sleep*, zsz180.
- **Estructura esperada:** cuatro carpetas `heart_rate`, `motion`, `steps`, `labels` (archivos por sujeto con identificador aleatorio).
- **Fecha de descarga:**

## Ojo
Es una sola noche con PSG por persona (más 7 días previos según el otro chat, sin verificar). Sirve para medir el error del reloj contra la
polisomnografía, no para entrenar riesgo de mala calidad de sueño. Apple Watch queda para una fase posterior.

## Estado de la descarga (revisado)
- Carpetas descargadas: `heart_rate` 31 archivos, `labels` 31, `motion` 6 de 31; **falta `steps`** (o aún no aparece).
- Los archivos de `motion` (aceleración cruda) son el grueso del peso; el ZIP oficial pesa 550 MB y la descarga archivo por archivo del directorio `files/` baja todo sin comprimir (2.2 GB).
- Etapas en `labels`: 0 (despierto), 1, 2, 3 y 5 (REM); no aparece la 4.
- Para nuestro objetivo no hace falta la aceleración: pulso, etapas y pasos alcanzan.
