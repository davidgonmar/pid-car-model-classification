import cv2
import os
from ultralytics import YOLO
import time



def crop_car_images(input_folder="input_folder", output_folder="data/processed"):
    """
    Recorta las imágenes para extraer solo los vehículos detectados por YOLOv8.
>>>>>>> de8424ac519a1e11208e91190d1bc9680bb9fbe1

    Args:
        input_folder (str): Carpeta con las imágenes de entrada.
        output_folder (str): Carpeta donde se guardarán las imágenes procesadas.
        ground_truth_folder (str): Carpeta con las imágenes de verdad de terreno (ground truth).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Cargar el modelo YOLO preentrenado
    model = YOLO("yolov8n.pt")

    # Inicialización de contadores
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_objects = 0
    start_time = time.time()

    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Hacer la detección con YOLO
            results = model(image)

            # Inicializar ground_truth_boxes como una lista vacía en caso de que no exista el archivo
            ground_truth_boxes = []

            # Cargar la ground truth si existe
            ground_truth_path = os.path.join(ground_truth_folder, filename.replace('.jpg', '.txt'))
            if os.path.exists(ground_truth_path):
                with open(ground_truth_path, 'r') as f:
                    ground_truth_boxes = [list(map(int, line.strip().split())) for line in f]

            for result in results:
                boxes = result.boxes.xyxy  # Bounding boxes (x_min, y_min, x_max, y_max)
                class_ids = result.boxes.cls  # Clases detectadas
                names = result.names

                for box, class_id in zip(boxes, class_ids):
                    class_name = names[int(class_id)]

                    # Filtrar solo vehículos
                    if class_name in ["car", "truck"]:
                        x_min, y_min, x_max, y_max = map(int, box)

                        # Aquí deberías calcular TP, FP y FN usando las ground truth
                        # Implementa una forma de comparar estas cajas

                        # Guardar la imagen recortada

                        output_path = os.path.join(
                            output_folder, f"{filename}_crop_{class_name}.jpg"
                        )
                        cv2.imwrite(output_path, vehicle_crop)

                        print(
                            f"Vehículo detectado: {class_name} - Guardado en {output_path}"
                        )

            # Contar cuántos objetos hay en la imagen (ground truth)
            total_objects += len(ground_truth_boxes)

    # Calcular FPS
    elapsed_time = time.time() - start_time
    fps = len(os.listdir(input_folder)) / elapsed_time

    # Aquí deberías calcular las métricas de precisión y exhaustividad
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    print(f"Precisión: {precision*100:.2f}%")
    print(f"Exhaustividad: {recall*100:.2f}%")
    print(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    evaluate_model()
