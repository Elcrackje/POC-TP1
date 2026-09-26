# Anonymized Wearable Sensor Health and Activity Dataset

- **Enlace:** https://zenodo.org/records/17967723
- **Tipo:** 2 (datos diarios, sin etiqueta externa)
- **Marca y modelo:** Garmin Vivosmart 5
- **Personas / días:** número de personas NO indicado. Recolección de septiembre 2024 a agosto 2025.
- **Variables (según la página):** frecuencia cardíaca (reposo, activa, variabilidad), métricas corporales (peso, IMC, "fitness age"),
  calidad de sueño (etapas y duración), actividad (pasos, minutos de intensidad, niveles de estrés).
- **Etiqueta:** ninguna. La "calidad de sueño" es la métrica propia de Garmin.
- **Población:** participantes ubicados en Grecia.
- **Licencia y descarga:** CC-BY 4.0. Descarga directa desde Zenodo.
- **Publicado:** 17 diciembre 2025 (v1). Autoría en la página: University of Patras; Tzamalis, Pantelis. Proyecto europeo SynAir-G.
- **Archivos esperados en esta carpeta:** `Garmin_Greece_202409-202508__SHA256.xlsx` (1.2 MB) y `Data Documentation.xlsx` (173 kB)
- **Fecha de descarga:**

## NO verificado / a revisar al abrir los archivos
- Cuántas personas hay y cuántos días tiene cada una (leer `Data Documentation.xlsx`).
- Si trae minutos despierto, eficiencia o latencia, y si los pasos son diarios.
- No hay paper asociado indicado.

## Al abrir el Excel (revisado)
- Hojas: `Sleep` (862 noches, 20 usuarios), `Activities` (279 sesiones, 22 usuarios), `Spo2` (41,931 filas, 27 usuarios).
- **Mediana de 4.5 noches por usuario; solo 7 usuarios tienen 10 noches o más.**
- **Población: alumnos de colegio.** Las columnas `school_id` (5 colegios) y `class24_25` (clases como E2, E1, G2, D1) lo indican, y el sueño medio es de 8.0 h con máximo de 11.2 h.
  Es muy probable que sean niños de primaria. No son universitarios.
- **No trae pulso en reposo ni HRV** (la página sí los menciona, pero no están en las hojas). Los pasos solo aparecen dentro de sesiones de actividad (167 filas).
- El archivo `Data Documentation.xlsx` es documentación general del proyecto europeo (hojas de otros datasets), no describe estos datos.
- **Uso realista:** no sirve para el modelo. Descartado, salvo como ejemplo de formato.
