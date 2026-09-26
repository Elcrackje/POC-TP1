# Multilevel Monitoring of Activity and Sleep in Healthy People (MMASH)

- **Enlace:** https://physionet.org/content/mmash/1.0.0/
- **Tipo:** 2 con etiqueta real (PSQI), pero con dispositivos de investigación
- **Marca y modelo:** Polar H7 (banda de pecho) y ActiGraph wGT3X-BT (actígrafo). NO son smartwatches comerciales.
- **Personas / duración:** 22 hombres adultos jóvenes sanos; 24 horas continuas (1 noche por persona).
- **Variables:** intervalos latido a latido, acelerometría triaxial, métricas de sueño, registro de actividad física, cortisol y melatonina en saliva.
- **Etiqueta:** cuestionarios MEQ, STAI-Y, **PSQI**, BIS/BAS, PANAS, DSI. El PSQI viene como **una sola puntuación global de 0 a 21**, sin ítems.
- **Licencia y descarga:** Open Data Commons Open Database License v1.0. Sin registro; ZIP directo (23.5 MB).
- **Publicado:** 19 junio 2020 (v1.0.0). Autores: Rossi et al.
- **Paper:** Rossi et al. (2020), *Data*, 5(4), 91. https://doi.org/10.3390/data5040091
- **Archivos por participante:** user_info.csv, sleep.csv, RR.csv, questionnaire.csv, Activity.csv, Actigraph.csv, saliva.csv
- **Fecha de descarga:**

## Ojo
- **No suma como marca** al modelo multi-wearable: nadie usa un Polar H7 con un ActiGraph. Sirve, como mucho, para una comprobación de sentido:
  ¿se relaciona el PSQI con las variables de sueño medidas por el actígrafo (duración, eficiencia, despertares)?
- n = 22, solo hombres y una noche, mientras el PSQI cubre el último mes: no permite entrenar ni validar un modelo, solo mirar tendencias.
- Descarga chica (23.5 MB); se puede bajar por si acaso. Prioridad baja.

## Al abrir los archivos (revisado)
- 22 participantes (carpeta `DataPaper/user_1` a `user_22`), edad media 26 (hay una edad en 0, dato erróneo), todos hombres.
- **PSQI global (columna `Pittsburgh`): de 2 a 9, mediana 5. 7 personas por encima de 5 y ninguna por encima de 10.**
- `sleep.csv` (del ActiGraph, una noche por persona): latencia, eficiencia, minutos en cama, tiempo total de sueño, WASO, número de despertares, índices de movimiento y fragmentación.
- Sirve para una comprobación de sentido: ¿el PSQI se relaciona con la eficiencia, el WASO y el tiempo de sueño medidos? Con n = 22 y una noche solo dará tendencias.
- Dato útil sobre los cortes: en 22 hombres jóvenes sanos, un corte de "Alto > 10" no habría marcado a nadie.
