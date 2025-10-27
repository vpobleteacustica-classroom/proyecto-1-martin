# Hito 2: Clasificador de Música por Género

**Integrantes:** Martin Arrigo

Este documento describe los avances del Hito 2 del proyecto, enfocado en la extracción de características y la implementación de un prototipo funcional de clasificación.

## 1. Avances Realizados

Se implementó con éxito la metodología definida en el Hito 1, logrando un prototipo funcional que incluye:

* **Extracción de Características:** Se implementó un script para procesar el dataset GTZAN. Se extraen 41 características acústicas (incluyendo MFCCs, Chroma, Contraste Espectral, ZCR y Tempo) de 1000 archivos de audio.
* **Gestión de Datos:** Las características extraídas se guardan y se cargan desde un archivo `features_extraidas.csv` para agilizar el análisis.
* **Verificación de Datos:** Se confirmó que el dataset está balanceado (100 canciones por género) y no contiene valores nulos.
* **Análisis Exploratorio (EDA):** Se generó un gráfico de cajas (boxplot) para visualizar la distribución del Tempo por género, identificando diferencias clave entre categorías.
* **Preprocesamiento:** Se implementó un pipeline de preprocesamiento que utiliza `StandardScaler` (para escalar las características) y `LabelEncoder` (para las etiquetas).
* **Modelo Base:** Se entrenó y evaluó un modelo KNN base, logrando una precisión inicial de ~57%, validando que las características extraídas son relevantes.

## 2. Mejoras de Código

Se modularizo el codigo:

* **`FP.py`:** Se creó un archivo auxiliar que contiene las funciones principales `extraer_caracteristicas` y `preprocesar_datos`.
* **`Segundo_Hito_Template.ipynb`:** El notebook se utiliza como informe principal, llamando a las funciones de `FP.py` para un análisis de los datos más limpio.

## 3. Pendiente para Hito 3

El trabajo para la entrega final (Hito 3) se centrará en la experimentación y optimización del modelo:

* **Probar otros modelos:** Entrenar y comparar el rendimiento de clasificadores más robustos, como Random Forest y Support Vector Machines (SVM).
* **Optimización:** Realizar ajuste de hiperparámetros (ej. `n_neighbors` en KNN o `n_estimators` en Random Forest) para mejorar la precisión.
* **Análisis de Resultados:** Realizar un análisis más profundo de la matriz de confusión para identificar qué géneros se confunden más y por qué.
* **Comparacion de modelos:** Evaluar el rendimiento de los modelos pendientes de entrenar (Random Forest, SVM) y contrastar todos los modelos para ver cuál es el mejor.

## 4. Pasos a seguir para ejecutar el codigo


Este registro contiene los archivos de audio del dataset GTZAN, utilizado para un proyecto del curso ACUS220 - Acústica Computacional con Python. El dataset original fue recopilado y descrito por Tzanetakis & Cook (2002).

Enlace del dataset usado:https://zenodo.org/uploads/17458653

Observación: Esta versión del dataset difiere del original en un archivo. El audio jazz.00054 (del GTZAN original) estaba corrupto y no podía ser procesado. Fue reemplazado por una pista de 30 segundos titulada "Jazz Background Music", obtenida de Pixabay (https://pixabay.com/es/music/search/jazz/). Esta pista fue recortada para coincidir con la duración estándar de 30 segundos de las muestras del dataset GTZAN.



### Dependencias

Se necesita tener instaladas las siguientes librerias de Python:


```bash
pip install pandas seaborn matplotlib scikit-learn librosa

Observación: En el primer paso se tiene modificar la ruta base para leer el dataset 