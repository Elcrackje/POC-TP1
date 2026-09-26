# Group Health (Sleep and Screen Time) Dataset

- **Enlace:** https://zenodo.org/records/15171250
- **Tipo:** 1 (reloj no-Fitbit + variable de calidad de sueño), con reservas (ver notas)
- **Marca y modelo:** Mi Band (Xiaomi). La página dice "Mi Band Smartwatches"; los identificadores de sesión traen "huami".
- **Personas / días:** número de personas NO indicado en la página. 301,556 registros durante "two weeks".
- **Variables (columnas):** Uid, Sid, Key, Time, Value, UpdateTime, screentime, expected_sleep, sleep_rating. Formato largo (clave-valor).
- **Etiqueta:** `sleep_rating`, "numerical rating of sleep quality" (ej. 0.65). NO es PSQI; no dice cómo se obtuvo.
- **Licencia y descarga:** CC-BY 4.0. Descarga directa desde Zenodo.
- **Publicado:** 8 abril 2025 (v1). Autoría en la página: Gogate; "created by students at the Eindhoven University of Technology".
- **Archivos esperados en esta carpeta:** `GroupHealth_SleepScreen.csv` (32.1 MB)
- **Fecha de descarga:**

## Verificado en la página (2026-09)
Título, licencia, archivo, marca (Mi Band), columnas, 301,556 registros, "two weeks", existencia de `sleep_rating`.

## NO verificado / a revisar al abrir el CSV
- Cuántas personas hay (contar valores distintos de `Uid`).
- Si los participantes son estudiantes: la página dice que el dataset lo crearon estudiantes de TU/e, no que los participantes lo sean.
- Cómo se calcula `sleep_rating` (¿autorreporte, o puntaje del reloj?). Si lo calcula el reloj, sirve poco como etiqueta independiente.
- Qué claves (`Key`) trae el formato largo: si incluye pulso, sueño y pasos diarios.
- No hay paper asociado indicado.

## Al abrir el CSV (revisado)
- **Solo 4 usuarios** (`Uid` 1 a 4), de 16 a 53 días cada uno (10 nov 2024 a 6 abr 2025). 301,556 filas, 6 dispositivos (`Sid`, prefijo "huami").
- Formato largo: cada fila trae en `Value` un JSON. Claves y filas: heart_rate 168,563 · abnormal_heart_rate 72,452 · calories 26,339 · steps 15,604 ·
  spo2 5,104 · stress 3,549 · single_heart_rate 2,121 · resting_heart_rate 151 · **sleep 138** · pai 84 · screentime 109.
- Cada registro `sleep` trae duración, minutos en sueño profundo, ligero, REM y despierto, hora de acostarse y de levantarse, pulso y SpO2 medios.
- `sleep_rating`: solo 59 valores (unos 15 por usuario, entre 0.0 y 0.9, mediana 0.7), todos en filas de `expected_sleep`. No se sabe cómo se obtiene.
- El pulso en reposo (`resting_heart_rate`) solo lo trae 1 de 4 usuarios (17 días); SpO2 (`spo2`) solo 1 usuario (47 días). El pulso por minuto sí está para todos.
- Minutos despierto promedian 6.6 y la eficiencia 98.6 % (contra 50.6 y 94.2 % en LifeSnaps): posible definición distinta de "despierto".
- **Uso realista:** no sirve para entrenar (n = 4). Sí sirve como **ejemplo real del formato de exportación de Xiaomi/Huami** para probar la capa de estandarización.
