import cv2
import os
from ultralytics import YOLO

def detect_and_draw(input_folder="input_folder", output_folder="data/processed"):
    """
    Dibuja un cuadro verde alrededor de los vehículos detectados, muestra el nombre y el porcentaje de confianza.
    
    Args:
        input_folder (str): Carpeta con las imágenes originales.
        output_folder (str): Carpeta donde se guardarán las imágenes procesadas.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Cargar el modelo YOLO preentrenado
    model = YOLO('yolov8n.pt')

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Hacer la detección con YOLO
            results = model(image)

            for result in results:
                boxes = result.boxes.xyxy  # Bounding boxes (x_min, y_min, x_max, y_max)
                class_ids = result.boxes.cls  # Clases detectadas
                confidences = result.boxes.conf  # Confianza de cada detección
                names = result.names

                for box, class_id, confidence in zip(boxes, class_ids, confidences):
                    class_name = names[int(class_id)]
                    confidence_pct = confidence * 100  # Convertir a porcentaje

                    # Filtrar solo vehículos
                    if class_name in ['car', 'truck']:
                        x_min, y_min, x_max, y_max = map(int, box)

                        # Dibujar rectángulo verde en la imagen
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        
                        # Escribir el nombre de la clase y la confianza
                        label = f"{class_name} ({confidence_pct:.2f}%)"
                        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.9, (0, 255, 0), 2)

            # Guardar la imagen con las detecciones
            output_path = os.path.join(output_folder, f"{filename}_detected.jpg")
            cv2.imwrite(output_path, image)
            print(f"Imagen procesada guardada en {output_path}")

    print("¡Procesamiento completado!")

if __name__ == "__main__":
    detect_and_draw()
