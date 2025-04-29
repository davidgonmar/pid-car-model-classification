import cv2
import os
from ultralytics import YOLO


def crop_car_images(input_folder="input_folder", output_folder="data/processed"):
    """
    Recorta las imágenes para extraer solo los vehículos detectados por YOLOv8.

    Args:
        input_folder (str): Carpeta con las imágenes originales.
        output_folder (str): Carpeta donde se guardarán las imágenes recortadas.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Cargar el modelo YOLO preentrenado
    model = YOLO("yolov8n.pt")

    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Hacer la detección con YOLO
            results = model(image)

            for result in results:
                boxes = result.boxes.xyxy  # Bounding boxes (x_min, y_min, x_max, y_max)
                class_ids = result.boxes.cls  # Clases detectadas
                names = result.names

                for box, class_id in zip(boxes, class_ids):
                    class_name = names[int(class_id)]

                    # Filtrar solo vehículos
                    if class_name in ["car", "truck"]:
                        x_min, y_min, x_max, y_max = map(int, box)

                        # Recortar la región del vehículo
                        vehicle_crop = image[y_min:y_max, x_min:x_max]

                        # Guardar la imagen recortada
                        output_path = os.path.join(
                            output_folder, f"{filename}_crop_{class_name}.jpg"
                        )
                        cv2.imwrite(output_path, vehicle_crop)

                        print(
                            f"Vehículo detectado: {class_name} - Guardado en {output_path}"
                        )

    print("¡Procesamiento completado!")


if __name__ == "__main__":
    crop_car_images()
