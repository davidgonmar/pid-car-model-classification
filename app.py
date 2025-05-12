import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as transforms
from lib import resnet
import json
import sys


@st.cache_resource(show_spinner=False)
def load_classes():
    classes_path = 'data/cars_dataset/meta.json'

    # Load class names from JSON file
    with open(classes_path, 'r') as f:
        class_meta = json.load(f)
        class_meta = class_meta['classes']
        print(f"Classes: {class_meta}")

    return class_meta

class_meta = load_classes()


@st.cache_resource(show_spinner=False)
def load_yolo():
    return YOLO('yolov8n.pt')
yolo_model = load_yolo()

# --- 2) Load ResNet50 (198-way) and weights ---
@st.cache_resource(show_spinner=False)
def load_resnet():
    models = {
        "50": resnet.ResNet50,
        "18": resnet.ResNet101,
    }
    assert len(sys.argv) == 3, "Usage: python app.py <model_name> <weights_path>"
    model_name = sys.argv[1]
    weights_path = sys.argv[2]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models[model_name](num_classes=198).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, device
resnet_model, device = load_resnet()

# --- 3) Preprocessing for ResNet ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def detect_and_classify(img: np.ndarray):
    """
    Run YOLO on the full image, then for each car/truck crop run ResNet.
    Returns (annotated_image, list_of_predicted_indices).
    """
    results = yolo_model.predict(source=img, conf=0.25, imgsz=640)[0]
    annotated = img.copy()
    preds = []
    for box in results.boxes:
        cls_id = int(box.cls.cpu().item())
        label = results.names[cls_id]
        if label not in ('car', 'truck'):
            continue
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Preprocess crop and classify
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = resnet_model(tensor)
            pred_idx = logits.argmax(1).item()
        preds.append(pred_idx)

        # Draw rectangle only (we'll show the number in the UI component)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return annotated, preds

def main():
    st.set_page_config(page_title="Vehicle Detector + Classifier")
    st.title("Vehicle Detector + Classifier")
    st.write("Upload a JPG/PNG to detect cars/trucks and classify them (indices 0–197).")

    uploaded = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Couldn't decode image.")
            return

        with st.spinner("Detecting and classifying..."):
            annotated, preds = detect_and_classify(img)

        # Show annotated image
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_container_width=True)

        # Show the predicted indices in a UI component
        if preds:
            with st.expander("View predicted class indices"):
                for i, p in enumerate(preds, start=1):
                    p = str(p)
                    nump = p
                    namep = class_meta[p]["name"]
                    srcp = class_meta[p]["src"]
                    str_ = f"Detected {namep} from source {srcp} with class index {nump}"
                    st.markdown(f"Detection {i}: {str_}")
        else:
            st.info("No cars or trucks detected.")

if __name__ == "__main__":
    main()
