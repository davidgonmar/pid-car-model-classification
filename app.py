import os
import shutil
import tempfile
import zipfile
import base64
from pathlib import Path
from io import BytesIO
import time

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from nicegui import ui
from PIL import Image
from ultralytics import YOLO

# Importar la función original
from utils.crop_and_resize import crop_car_images

# Configuración de carpetas
INPUT_FOLDER = "input_folder"
OUTPUT_FOLDER = "data/processed"

# Asegurar que las carpetas existan
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Cargar el modelo YOLO (igual que en crop_car_images)
try:
    yolo_model = YOLO("yolov8n.pt")
    print("Modelo YOLO cargado correctamente")
except Exception as e:
    print(f"Error al cargar el modelo YOLO: {e}")
    yolo_model = None

# Cargar el modelo ResNet50 preentrenado
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet_model.eval()
    resnet_model.to(device)
    
    # Preprocesamiento para ResNet50
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
except Exception as e:
    print(f"Error al cargar el modelo ResNet: {e}")
    resnet_model = None

def detect_vehicles_with_boxes(image_path):
    """Detecta vehículos en una imagen usando YOLOv8 y devuelve imagen con bounding boxes"""
    if yolo_model is None:
        print("Modelo YOLO no disponible")
        return None, []
    
    # Leer la imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error al leer la imagen: {image_path}")
        return None, []
    
    # Hacer la detección con YOLO (igual que en crop_car_images)
    results = yolo_model(image)
    
    # Crear una copia de la imagen para dibujar los bounding boxes
    annotated_img = image.copy()
    
    # Lista para almacenar información de los vehículos detectados
    detected_vehicles = []
    
    vehicle_count = 0
    
    for result in results:
        boxes = result.boxes.xyxy  # Bounding boxes (x_min, y_min, x_max, y_max)
        class_ids = result.boxes.cls  # Clases detectadas
        conf_scores = result.boxes.conf  # Puntuaciones de confianza
        names = result.names
        
        print(f"Detecciones totales: {len(boxes)}")
        
        for box, class_id, conf in zip(boxes, class_ids, conf_scores):
            class_name = names[int(class_id)]
            confidence = float(conf) * 100
            
            print(f"Clase detectada: {class_name}, Confianza: {confidence:.2f}%")
            
            # Filtrar solo vehículos (igual que en crop_car_images)
            if class_name in ["car", "truck"]:
                vehicle_count += 1
                x_min, y_min, x_max, y_max = map(int, box)
                
                # Dibujar el bounding box
                cv2.rectangle(annotated_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                label = f"{class_name} {confidence:.1f}%"
                cv2.putText(annotated_img, label, (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Guardar información del vehículo
                vehicle_info = {
                    "class_name": class_name,
                    "confidence": confidence,
                    "box": (x_min, y_min, x_max, y_max),
                    "output_path": os.path.join(
                        OUTPUT_FOLDER, f"{os.path.basename(image_path)}_crop_{class_name}_{vehicle_count}.jpg"
                    )
                }
                detected_vehicles.append(vehicle_info)
    
    print(f"Vehículos detectados: {vehicle_count}")
    return annotated_img, detected_vehicles

def classify_image(image_path):
    """Clasifica una imagen con ResNet50"""
    if resnet_model is None:
        return [
            {'category': 1, 'probability': 95.5},
            {'category': 2, 'probability': 3.2},
            {'category': 3, 'probability': 1.0},
            {'category': 4, 'probability': 0.2},
            {'category': 5, 'probability': 0.1}
        ]
    
    try:
        # Abrir y preprocesar la imagen
        image = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = resnet_model(input_tensor)
        
        # Obtener las probabilidades
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Obtener las 5 predicciones principales
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        
        results = []
        for i in range(5):
            results.append({
                'category': int(top5_catid[i]),  # Usar el número de clase directamente
                'probability': float(top5_prob[i]) * 100
            })
        
        return results
    except Exception as e:
        print(f"Error al clasificar imagen: {e}")
        return [
            {'category': 0, 'probability': 100.0},
            {'category': 0, 'probability': 0.0},
            {'category': 0, 'probability': 0.0},
            {'category': 0, 'probability': 0.0},
            {'category': 0, 'probability': 0.0}
        ]

def cv2_to_base64(img):
    """Convierte una imagen de OpenCV a base64 para mostrarla en HTML"""
    _, buffer = cv2.imencode('.jpg', img)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

# Clase para manejar la interfaz de usuario
class CarDetectionApp:
    def __init__(self):
        self.detected_vehicles = []
        self.setup_ui()

    def setup_ui(self):
        # Título y descripción
        with ui.card().classes('w-full'):
            ui.label('Detección y Clasificación de Vehículos').classes('text-2xl font-bold')
            ui.label('Sube imágenes, detecta vehículos y clasifícalos con ResNet50').classes('text-gray-500')
        
        # Sección de carga de archivos
        with ui.card().classes('w-full mt-4'):
            ui.label('1. Sube imágenes o un archivo ZIP').classes('text-xl font-bold')
            
            with ui.row().classes('w-full justify-between items-center'):
                with ui.column().classes('w-1/2'):
                    self.upload = ui.upload(
                        label='Arrastra archivos aquí o haz clic para seleccionar',
                        multiple=True,
                        on_upload=self.handle_upload
                    ).props('accept=".jpg,.jpeg,.png,.zip"').classes('w-full')
                
                with ui.column().classes('w-1/2 pl-4'):
                    ui.button('Limpiar carpeta de entrada', on_click=self.clear_input_folder).classes('bg-red-500 text-white')
        
        # Sección de procesamiento por lotes
        with ui.card().classes('w-full mt-4'):
            ui.label('2. Procesar todas las imágenes').classes('text-xl font-bold')
            
            with ui.row().classes('w-full'):
                ui.button('Detectar todos los vehículos', on_click=self.process_all_images).classes('bg-blue-500 text-white')
                self.status_label = ui.label('Esperando procesamiento...').classes('ml-4')
        
        # Sección de imágenes individuales
        with ui.card().classes('w-full mt-4'):
            ui.label('3. Imágenes disponibles').classes('text-xl font-bold')
            self.images_container = ui.element('div').classes('w-full grid grid-cols-3 gap-4 mt-4')
        
        # Sección de detección
        with ui.card().classes('w-full mt-4'):
            ui.label('4. Visualización de detección').classes('text-xl font-bold')
            self.detection_container = ui.element('div').classes('w-full mt-4')
        
        # Sección de recortes
        with ui.card().classes('w-full mt-4'):
            ui.label('5. Vehículos detectados').classes('text-xl font-bold')
            self.crops_container = ui.element('div').classes('w-full grid grid-cols-3 gap-4 mt-4')
        
        # Sección de clasificación
        with ui.card().classes('w-full mt-4'):
            ui.label('6. Resultados de clasificación').classes('text-xl font-bold')
            self.classification_container = ui.element('div').classes('w-full')

    def handle_upload(self, e):
        """Maneja la carga de archivos a la carpeta de entrada"""
        for file in e.files:
            file_path = os.path.join(INPUT_FOLDER, file.name)
            
            # Guardar el archivo
            with open(file_path, 'wb') as f:
                f.write(file.content.read())
            
            # Si es un ZIP, descomprimir
            if file.name.lower().endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(INPUT_FOLDER)
                # Eliminar el ZIP después de descomprimir
                os.remove(file_path)
        
        ui.notify(f'Archivos subidos correctamente a {INPUT_FOLDER}')
        self.update_images_list()

    def clear_input_folder(self):
        """Limpia la carpeta de entrada"""
        for file in os.listdir(INPUT_FOLDER):
            file_path = os.path.join(INPUT_FOLDER, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        ui.notify('Carpeta de entrada limpiada')
        self.update_images_list()
        
        # También limpiar la carpeta de salida
        if os.path.exists(OUTPUT_FOLDER):
            for file in os.listdir(OUTPUT_FOLDER):
                file_path = os.path.join(OUTPUT_FOLDER, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    def process_all_images(self):
        """Procesa todas las imágenes usando la función crop_car_images original"""
        self.status_label.text = 'Procesando imágenes...'
        
        # Limpiar la carpeta de salida
        if os.path.exists(OUTPUT_FOLDER):
            for file in os.listdir(OUTPUT_FOLDER):
                file_path = os.path.join(OUTPUT_FOLDER, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        # Llamar a la función original de crop_car_images
        try:
            crop_car_images(input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER)
            self.status_label.text = 'Procesamiento completado'
            self.update_crops_list()
        except Exception as e:
            self.status_label.text = f'Error: {str(e)}'
            ui.notify(f'Error al procesar imágenes: {str(e)}', type='negative')

    def update_images_list(self):
        """Actualiza la lista de imágenes disponibles"""
        self.images_container.clear()
        
        image_files = []
        for file in os.listdir(INPUT_FOLDER):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(INPUT_FOLDER, file))
        
        if not image_files:
            with self.images_container:
                ui.label('No hay imágenes disponibles').classes('col-span-3 text-center text-gray-500')
        else:
            for img_path in image_files:
                with self.images_container:
                    with ui.card().classes('w-full'):
                        ui.image(img_path).classes('w-full h-40 object-cover')
                        with ui.row().classes('w-full justify-between items-center'):
                            ui.label(os.path.basename(img_path)).classes('text-xs truncate')
                            ui.button('Visualizar detección', on_click=lambda p=img_path: self.visualize_detection(p)).classes('bg-blue-500 text-white text-xs')

    def update_crops_list(self):
        """Actualiza la lista de recortes disponibles"""
        self.crops_container.clear()
        
        if not os.path.exists(OUTPUT_FOLDER):
            with self.crops_container:
                ui.label('No hay recortes disponibles').classes('col-span-3 text-center text-gray-500')
            return
        
        crop_files = []
        for file in os.listdir(OUTPUT_FOLDER):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                crop_files.append(os.path.join(OUTPUT_FOLDER, file))
        
        if not crop_files:
            with self.crops_container:
                ui.label('No hay recortes disponibles').classes('col-span-3 text-center text-gray-500')
        else:
            for crop_path in crop_files:
                with self.crops_container:
                    with ui.card().classes('w-full'):
                        ui.image(crop_path).classes('w-full h-40 object-cover')
                        with ui.row().classes('w-full justify-between items-center'):
                            ui.label(os.path.basename(crop_path)).classes('text-xs truncate')
                            ui.button('Clasificar', on_click=lambda p=crop_path: self.show_classification(p)).classes('bg-green-500 text-white text-xs')

    def visualize_detection(self, image_path):
        """Visualiza la detección de vehículos en una imagen"""
        self.detection_container.clear()
        
        # Mostrar mensaje de procesamiento
        with self.detection_container:
            processing_label = ui.label('Procesando imagen...').classes('text-blue-500')
        
        # Detectar vehículos y obtener imagen con bounding boxes
        annotated_img, detected_vehicles = detect_vehicles_with_boxes(image_path)
        self.detected_vehicles = detected_vehicles
        
        # Limpiar el contenedor de detección
        self.detection_container.clear()
        
        if annotated_img is None:
            with self.detection_container:
                ui.label('Error al procesar la imagen').classes('text-red-500')
            return
        
        # Mostrar imagen con bounding boxes
        with self.detection_container:
            ui.label(f'Imagen procesada: {os.path.basename(image_path)}').classes('text-lg font-bold')
            
            # Convertir la imagen a base64 para mostrarla
            img_base64 = cv2_to_base64(annotated_img)
            ui.html(f'<img src="{img_base64}" class="w-full max-h-96 object-contain">')
            
            if len(detected_vehicles) > 0:
                ui.label(f'Vehículos detectados: {len(detected_vehicles)}').classes('mt-2 text-green-600 font-bold')
                
                # Mostrar detalles de cada vehículo detectado
                with ui.element('div').classes('mt-2 p-2 bg-gray-100 rounded'):
                    for i, vehicle in enumerate(detected_vehicles):
                        ui.label(f"Vehículo #{i+1}: {vehicle['class_name']} - Confianza: {vehicle['confidence']:.2f}%")
            else:
                ui.label('No se detectaron vehículos en esta imagen').classes('mt-2 text-orange-500')
            
            # Botón para procesar esta imagen con crop_car_images
            ui.button('Guardar recortes de esta imagen', on_click=lambda: self.process_single_image(image_path)).classes('bg-blue-500 text-white mt-2')

    def process_single_image(self, image_path):
        """Procesa una sola imagen con la función crop_car_images"""
        # Crear una carpeta temporal para la imagen
        temp_folder = os.path.join(INPUT_FOLDER, "temp_single")
        os.makedirs(temp_folder, exist_ok=True)
        
        # Copiar la imagen a la carpeta temporal
        temp_image_path = os.path.join(temp_folder, os.path.basename(image_path))
        shutil.copy(image_path, temp_image_path)
        
        try:
            # Llamar a crop_car_images con la carpeta temporal
            crop_car_images(input_folder=temp_folder, output_folder=OUTPUT_FOLDER)
            ui.notify('Recortes guardados correctamente')
            self.update_crops_list()
        except Exception as e:
            ui.notify(f'Error al procesar la imagen: {str(e)}', type='negative')
        finally:
            # Limpiar la carpeta temporal
            shutil.rmtree(temp_folder, ignore_errors=True)

    def show_classification(self, image_path):
        """Muestra los resultados de clasificación para una imagen"""
        self.classification_container.clear()
        
        # Mostrar mensaje de procesamiento
        with self.classification_container:
            processing_label = ui.label('Clasificando imagen...').classes('text-blue-500')
        
        # Clasificar la imagen
        results = classify_image(image_path)
        
        # Limpiar el contenedor de clasificación
        self.classification_container.clear()
        
        with self.classification_container:
            with ui.row().classes('w-full'):
                with ui.column().classes('w-1/3'):
                    ui.image(image_path).classes('w-full object-contain')
                    ui.label(os.path.basename(image_path)).classes('text-center')
                
                with ui.column().classes('w-2/3'):
                    ui.label('Resultados de clasificación ResNet50:').classes('text-lg font-bold')
                    
                    # Mostrar resultados
                    with ui.element('div').classes('w-full'):
                        for result in results:
                            with ui.row().classes('w-full items-center'):
                                ui.label(f"Clase {result['category']}:").classes('w-2/3')
                                with ui.element('div').classes('w-1/3 bg-gray-200 rounded-full h-4'):
                                    ui.element('div').classes(f'bg-blue-600 h-4 rounded-full').style(f"width: {result['probability']}%")
                                ui.label(f"{result['probability']:.2f}%").classes('ml-2')

# Iniciar la aplicación directamente
print("Iniciando servidor NiceGUI...")
print("Abre tu navegador y visita: http://localhost:8080")

# Crear la instancia de la aplicación
app_instance = CarDetectionApp()

# Ejecutar la aplicación
ui.run(title='Detector y Clasificador de Vehículos', favicon='🚗', port=8080) 