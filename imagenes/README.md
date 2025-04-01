# Visualizaciones para Paper de Clasificación de Vehículos

Este directorio contiene scripts para generar visualizaciones que complementan el paper sobre clasificación de vehículos basada en deep learning.

## Visualizaciones Disponibles

1. **Arquitectura de CNN (ResNet50)**
   - Archivo: `resnet50_architecture.jpg`
   - Descripción: Diagrama esquemático de la arquitectura ResNet50 utilizada como base para la clasificación de vehículos.

2. **Comparación de Modelos**
   - Archivo: `model_comparison.jpg`
   - Descripción: Gráfico comparativo de diferentes arquitecturas CNN evaluando precisión vs. tiempo de inferencia, con tamaño proporcional al número de parámetros.

3. **Visualizaciones Grad-CAM**
   - Archivo: `gradcam_comparison.jpg`
   - Descripción: Comparación de mapas de activación para diferentes modelos, mostrando qué áreas de la imagen son más relevantes para la clasificación.

4. **Data Augmentation**
   - Archivos: `data_augmentation_examples.jpg`, `data_augmentation_impact.jpg`
   - Descripción: Ejemplos visuales de técnicas de data augmentation y un gráfico que muestra su impacto en la precisión y robustez del modelo.

5. **Matriz de Confusión**
   - Archivo: `confusion_matrix.jpg`
   - Descripción: Visualización de la matriz de confusión del mejor modelo (ResNet50), mostrando patrones de errores de clasificación.

## Cómo Generar las Visualizaciones

Para generar todas las visualizaciones a la vez, ejecute:

```bash
python generate_all_plots.py
```

Para generar visualizaciones individuales, ejecute cualquiera de los scripts específicos:

```bash
python model_architecture.py
python model_comparison.py
python gradcam_visualization.py
python data_augmentation.py
python confusion_matrix.py
```

## Requisitos

Los scripts requieren las siguientes bibliotecas de Python:
- matplotlib
- numpy
- scipy
- seaborn

Puede instalarlas mediante:

```bash
pip install matplotlib numpy scipy seaborn
```

## Uso en el Paper

Estas visualizaciones se han diseñado para complementar el análisis presentado en el paper. Cada imagen puede insertarse en la sección correspondiente del documento LaTeX utilizando:

```latex
\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{imagenes/nombre_de_archivo.jpg}
\caption{Descripción de la figura.}
\label{fig:etiqueta}
\end{figure}
```

## Personalización

Los scripts están diseñados para ser fácilmente modificables en caso de que se necesiten ajustes específicos:

- Colores y estilos
- Etiquetas y anotaciones
- Configuración de parámetros y valores de datos

Para realizar modificaciones, edite los archivos Python correspondientes y vuelva a ejecutarlos. 