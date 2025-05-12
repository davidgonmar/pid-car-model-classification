#!/usr/bin/env python
import cv2
import os
import sys
import argparse
from PIL import Image
import numpy as np

# Añadir el directorio raíz del proyecto al sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

# Intentar cargar YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: No se pudo importar YOLO. Instalando...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        from ultralytics import YOLO
    except Exception as e:
        print(f"Error al instalar YOLO: {e}")
        sys.exit(1)

def crop_car_images(input_folder, output_folder, yolo_model_path=None):
    """
    Recorta las imágenes para extraer solo los vehículos detectados por YOLO.
    
    Args:
        input_folder (str): Carpeta con las imágenes de entrada.
        output_folder (str): Carpeta donde se guardarán las imágenes procesadas.
        yolo_model_path (str): Ruta al modelo YOLO preentrenado.
    """
    # Asegurarse de que las carpetas existan
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Determinar ruta al modelo YOLO
    if yolo_model_path is None:
        yolo_model_path = os.path.join(project_root, "utils", "yolov8n.pt")
        if not os.path.exists(yolo_model_path):
            yolo_model_path = "yolov8n"  # Usar modelo por defecto si no encontramos uno local
    
    print(f"Usando modelo YOLO: {yolo_model_path}")
    
    # Cargar el modelo YOLO preentrenado
    try:
        model = YOLO(yolo_model_path)
    except Exception as e:
        print(f"Error al cargar modelo YOLO: {e}")
        return
    
    processed_count = 0
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Error al cargar la imagen: {image_path}")
                continue
            
            # Hacer la detección con YOLO
            try:
                results = model(image)
                
                # Verificar si se detectaron objetos
                vehicle_found = False
                
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes (x_min, y_min, x_max, y_max)
                    class_ids = result.boxes.cls.cpu().numpy()  # Clases detectadas
                    names = result.names  # Diccionario de nombres de clases
                    
                    for box, class_id in zip(boxes, class_ids):
                        class_name = names[int(class_id)]
                        
                        # Filtrar solo vehículos
                        if class_name in ["car", "truck", "bus", "motorcycle"]:
                            x_min, y_min, x_max, y_max = map(int, box)
                            
                            # Recortar la imagen
                            vehicle_crop = image[y_min:y_max, x_min:x_max]
                            
                            # Redimensionar a 224x224 (tamaño estándar para ResNet)
                            resized_image = cv2.resize(vehicle_crop, (224, 224))
                            
                            # Guardar la imagen recortada
                            output_path = os.path.join(
                                output_folder, f"{filename}_crop_{class_name}.jpg"
                            )
                            cv2.imwrite(output_path, resized_image)
                            
                            print(f"Vehículo detectado: {class_name} - Guardado en {output_path}")
                            vehicle_found = True
                            processed_count += 1
                            
                            # Solo procesamos un vehículo por imagen para simplificar
                            break
                    
                    if vehicle_found:
                        break
                
                # Si no se encontró ningún vehículo, simplemente redimensionar la imagen
                if not vehicle_found:
                    print(f"No se detectaron vehículos en {filename}. Redimensionando.")
                    resized_image = cv2.resize(image, (224, 224))
                    output_path = os.path.join(output_folder, f"{filename}_crop_car.jpg")
                    cv2.imwrite(output_path, resized_image)
                    processed_count += 1
                    
            except Exception as e:
                print(f"Error al procesar la imagen {filename}: {e}")
                # En caso de error, redimensionar la imagen original
                try:
                    resized_image = cv2.resize(image, (224, 224))
                    output_path = os.path.join(output_folder, f"{filename}_crop_car.jpg")
                    cv2.imwrite(output_path, resized_image)
                    print(f"Error al detectar. Imagen redimensionada guardada en {output_path}")
                    processed_count += 1
                except Exception as resize_error:
                    print(f"Error al redimensionar: {resize_error}")
    
    return processed_count

def main():
    parser = argparse.ArgumentParser(description='Recortar y redimensionar imágenes de coches')
    parser.add_argument('--input', type=str, default='input_folder', help='Carpeta de entrada con imágenes')
    parser.add_argument('--output', type=str, default='output_folder', help='Carpeta de salida para imágenes procesadas')
    parser.add_argument('--yolo-model', type=str, default=None, help='Ruta al modelo YOLO (opcional)')
    
    args = parser.parse_args()
    
    processed_count = crop_car_images(args.input, args.output, args.yolo_model)
    print(f"Total de imágenes procesadas: {processed_count}")

if __name__ == "__main__":
    main() 