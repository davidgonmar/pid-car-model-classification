import cv2
import os
from ultralytics import YOLO
import time


def crop_car_images(input_folder="input_folder", output_folder="data/processed"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model = YOLO("yolov8n.pt")

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_objects = 0
    start_time = time.time()

    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Error al leer la imagen: {image_path}")
                continue

            results = model(image)

            ground_truth_boxes = []

            for result in results:
                boxes = result.boxes.xyxy
                class_ids = result.boxes.cls
                names = result.names

                for box, class_id in zip(boxes, class_ids):
                    class_name = names[int(class_id)]

                    if class_name in ["car", "truck"]:
                        x_min, y_min, x_max, y_max = map(int, box)
                        
                        vehicle_crop = image[y_min:y_max, x_min:x_max]
                        
                        output_path = os.path.join(
                            output_folder, f"{filename}_crop_{class_name}.jpg"
                        )
                        cv2.imwrite(output_path, vehicle_crop)

                        print(
                            f"Vehículo detectado: {class_name} - Guardado en {output_path}"
                        )

            total_objects += len(ground_truth_boxes)

    elapsed_time = time.time() - start_time
    fps = len(os.listdir(input_folder)) / elapsed_time if elapsed_time > 0 else 0

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    print(f"Precisión: {precision*100:.2f}%")
    print(f"Exhaustividad: {recall*100:.2f}%")
    print(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    crop_car_images()
