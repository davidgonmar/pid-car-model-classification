# Car Model Classifier

Una aplicación web Next.js que puede clasificar modelos de coches utilizando redes neuronales ResNet.

## Características

- Subir imágenes de coches a través de una interfaz amigable
- Recorte y redimensionamiento automático de imágenes para prepararlas para el modelo
- Seleccionar entre múltiples arquitecturas ResNet (ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152)
- Mostrar resultados de clasificación con puntuaciones de confianza

## Tecnologías Utilizadas

- Next.js 14 con App Router
- React 18
- Tailwind CSS
- Componentes HeadlessUI
- React Dropzone para cargas de archivos
- API Canvas para procesamiento de imágenes
- Python para ejecutar los modelos de clasificación reales

## Requisitos

Para usar la funcionalidad completa con los scripts de Python, necesitarás:

- Node.js 18.17 o superior
- Python 3.8 o superior
- PyTorch y Torchvision
- Ultralytics (para YOLOv8)
- OpenCV

## Instalación

1. Clona el repositorio:
   ```bash
   git clone <url-del-repositorio>
   cd car-model-classifier
   ```

2. Instala las dependencias de Node.js:
   ```bash
   npm install
   ```

3. Instala las dependencias de Python:
   ```bash
   pip install -r ../requirements.txt
   pip install ultralytics opencv-python
   ```

4. Ejecuta el servidor de desarrollo:
   ```bash
   npm run dev
   ```

5. Abre [http://localhost:3000](http://localhost:3000) en tu navegador.

## Integración con Scripts de Python

Esta aplicación se integra con los scripts de Python del repositorio principal para:

1. Recortar y redimensionar imágenes utilizando YOLOv8 para detectar vehículos
2. Clasificar el modelo del coche utilizando redes ResNet

Los scripts se han adaptado y se encuentran en `src/server/`:
- `crop_and_resize.py`: Script para detectar y recortar vehículos en imágenes
- `test_resnet.py`: Script para clasificar el modelo del coche

## Estructura del Proyecto

```
car-model-classifier/
├── public/
├── src/
│   ├── app/
│   │   ├── api/
│   │   │   └── predict/
│   │   │       └── route.js     # Endpoint API que llama a los scripts de Python
│   │   ├── classify/
│   │   │   └── page.js          # Página de clasificación de coches
│   │   ├── layout.js            # Layout raíz
│   │   └── page.js              # Página de inicio
│   ├── server/
│   │   ├── crop_and_resize.py   # Script adaptado para recortar y redimensionar
│   │   └── test_resnet.py       # Script adaptado para clasificación con ResNet
│   ├── utils/
│   │   ├── cropAndResize.js     # Utilidad de procesamiento de imágenes para el navegador
│   │   └── models.js            # Información de modelos ResNet
├── temp/
│   ├── input/                   # Directorio temporal para imágenes de entrada
│   └── output/                  # Directorio temporal para imágenes procesadas
```

## Cómo Funciona la Integración con Python

1. El usuario sube una imagen desde la interfaz web
2. La imagen se envía al endpoint de API (`/api/predict`)
3. La API guarda la imagen en un directorio temporal
4. La API ejecuta el script `crop_and_resize.py` para detectar y recortar el vehículo
5. La API ejecuta el script `test_resnet.py` con el modelo ResNet seleccionado
6. Los resultados se devuelven al frontend y se muestran al usuario

## Solución de Problemas

Si encuentras problemas con la ejecución de los scripts de Python:

1. Asegúrate de tener todas las dependencias de Python instaladas
2. Verifica que los scripts tengan permisos de ejecución
3. Comprueba las rutas a los scripts y directorios temporales
4. Revisa los logs del servidor para ver mensajes de error detallados

## Mejoras Futuras

- Mejora de la interfaz de usuario para mostrar más detalles sobre los modelos
- Soporte para más arquitecturas de modelos
- Entrenamiento personalizado en conjuntos de datos específicos de coches
- Optimización para móviles
