# Deteccion_Fenomenos_Meteorologicos_Extremos

Código desarrollado para la detección y análisis de fenómenos meteorológicos extremos en España, integrando datos de reanálisis, satélite y calidad del aire.

El repositorio incluye los pipelines de extracción, limpieza, preparación de datasets, modelado predictivo, explicabilidad y visualización empleados en el trabajo.

## Estructura del repositorio

```text
src/
├── extraccion/     # Descarga de datos desde APIs y servicios externos
├── limpieza/       # Limpieza, homogeneización y procesado de variables
├── modeling/       # Preparación de datasets, entrenamiento y evaluación de modelos
└── dashboard/      # Aplicación de visualización interactiva

notebooks/
├── 01_extraccion_datos.ipynb
├── 02_limpieza_y_procesado.ipynb
├── 03_dataset_olas_calor.ipynb
├── 04_modelado_olas_calor.ipynb
├── 05_dataset_precipitaciones.ipynb
├── 06_modelado_precipitaciones.ipynb
└── 07_explicabilidad_y_mapas.ipynb

data/
└── datos_tfg/      # Carpeta local para datos, no incluida en el repositorio
```

## Descripción general

El proyecto se organiza en torno a un flujo completo de trabajo para la detección de fenómenos meteorológicos extremos:

1. Extracción de datos meteorológicos, atmosféricos y satelitales.
2. Limpieza y homogeneización espacial y temporal de las variables.
3. Construcción de datasets para modelado.
4. Entrenamiento de modelos predictivos.
5. Evaluación de resultados y explicabilidad.
6. Visualización mediante mapas, gráficos y dashboard.

El objetivo es centralizar el código utilizado durante el trabajo, evitando depender de notebooks largos con funciones duplicadas.

## Datos

Los datos no se incluyen en el repositorio debido a su tamaño.

El código espera que los archivos se organicen localmente dentro de:

```text
data/datos_tfg/
```

Esta carpeta puede contener datos procedentes de ERA5, CAMS, Sentinel-5P, MODIS, GPM y productos derivados generados durante las fases de limpieza y procesado.

## Uso de los notebooks

Los notebooks están numerados siguiendo el flujo del proyecto:

```text
01_extraccion_datos.ipynb
```

Permite lanzar el pipeline de descarga de datos.

```text
02_limpieza_y_procesado.ipynb
```

Ejecuta las funciones de limpieza, homogeneización y generación de productos limpios.

```text
03_dataset_olas_calor.ipynb
```

Prepara el dataset tabular para el modelado de olas de calor.

```text
04_modelado_olas_calor.ipynb
```

Entrena y evalúa modelos para detección de olas de calor.

```text
05_dataset_precipitaciones.ipynb
```

Prepara datasets para precipitación extrema, incluyendo enfoques tabulares y secuenciales.

```text
06_modelado_precipitaciones.ipynb
```

Entrena modelos para precipitación extrema, incluyendo modelos tabulares y redes secuenciales.

```text
07_explicabilidad_y_mapas.ipynb
```

Incluye funciones de explicabilidad, análisis SHAP y visualización espacial de resultados.

## Módulos principales

```text
src/extraccion/
```

Contiene el pipeline de extracción de datos desde APIs y servicios externos.

```text
src/limpieza/
```

Contiene funciones de limpieza de datos, procesado de productos satelitales y construcción de cubos de datos limpios.

```text
src/modeling/
```

Contiene funciones de preparación de datasets, pipelines de entrenamiento, evaluación de modelos, explicabilidad y visualización de resultados.

```text
src/dashboard/
```

Contiene el código asociado al dashboard interactivo de exploración y visualización.

## Nota

Este repositorio forma parte de un Trabajo de Fin de Grado orientado al análisis y predicción de fenómenos meteorológicos extremos mediante técnicas de aprendizaje automático.
