# Hito 3: Clasificador de Música por Género


**Asignatura:** Acústica Computacional 

**Integrantes:** Martin Arrigo





##  Descripción

Este proyecto constituye la entrega final (Hito 3) para el desarrollo de un sistema automático de clasificación de géneros musicales. Utilizando el dataset **GTZAN**, el objetivo principal fue evolucionar desde una línea base (establecida en el Hito 2) hacia modelos más robustos, implementando optimización de hiperparámetros y comparativas métricas rigurosas.

El flujo de trabajo abarca desde la extracción de características de audio hasta la evaluación de modelos de **Machine Learning**.

## Objetivos

1.  **Extracción de Características:** Procesamiento de audio digital para obtener descriptores espectrales y temporales.
2.  **Experimentación:** Implementación de algoritmos KNN, Random Forest y Support Vector Machines (SVM).
3.  **Optimización:** Ajuste fino utilizando `GridSearchCV` para maximizar el rendimiento.
4.  **Comparación:** Análisis de métricas (Precision, Recall, F1-Score) para seleccionar el modelo óptimo.

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

```text
├── Data/                   # Dataset de audio (GTZAN)
├── figuras/                # Imágenes y logos (e.g., EscudoUACH.png)
├── info_generada/          # Archivos CSV procesados (features_extraidas.csv)
├── src/
│   └── FP.py               # Script de funciones (extracción y preprocesamiento)
├── notebooks/
│   └── Hito3_versionFinal.ipynb  # Notebook principal de ejecución
└── README.md               # Documentación del proyecto