from kaggle.api.kaggle_api_extended import KaggleApi
import scipy.io
import os
import shutil
import numpy as np
from collections import defaultdict
from PIL import Image
import json
from ultralytics import YOLO  # 🆕 Added for YOLO detection

# Paths
prjctroot = os.getcwd()
root = os.path.join(prjctroot, "data")
images_dir = os.path.join(prjctroot, "dataset-additional")
dataset_dir = os.path.join(root, "cars_dataset")

# Clean previous dataset
if os.path.exists(dataset_dir):
    shutil.rmtree(dataset_dir)

# Kaggle API Authentication
api = KaggleApi()
api.authenticate()


# Load Stanford Cars Dataset
def get_stanford_cars(split: str):
    stanford_dir = os.path.join(root, "stanford_cars")
    if not os.path.exists(stanford_dir):
        os.makedirs(stanford_dir, exist_ok=True)
        api.dataset_download_files(
            "rickyyyyyyy/torchvision-stanford-cars", path=root, unzip=True
        )
    if split == "test":
        mat_path = os.path.join(stanford_dir, "cars_test_annos_withlabels.mat")
        imgpath = os.path.join(stanford_dir, "cars_test")
    else:
        mat_path = os.path.join(stanford_dir, "devkit", "cars_train_annos.mat")
        imgpath = os.path.join(stanford_dir, "cars_train")
    data = scipy.io.loadmat(mat_path)
    annos = data["annotations"][0]
    samples = []
    for a in annos:
        bbox = [int(a[0][0]), int(a[1][0]), int(a[2][0]), int(a[3][0])]
        cls = int(a[4][0]) - 1
        fname = a[5][0]
        samples.append({"bbox": bbox, "class": cls, "fname": fname})
    return samples, imgpath


# Create folders
for split in ["train", "test"]:
    os.makedirs(os.path.join(dataset_dir, split, "images"), exist_ok=True)

train_samples, train_imgpath = get_stanford_cars("train")
test_samples, test_imgpath = get_stanford_cars("test")
print("✅ Stanford Cars dataset loaded successfully!")

# Process Stanford Cars training images
train_labels = []
class_counts_train = defaultdict(int)
for sample in train_samples:
    c = sample["class"]
    dst_dir = os.path.join(dataset_dir, "train", "images", str(c))
    os.makedirs(dst_dir, exist_ok=True)
    class_counts_train[c] += 1
    new_name = str(class_counts_train[c]).zfill(5) + ".jpg"
    src = os.path.join(train_imgpath, sample["fname"])
    img = Image.open(src)
    x1, y1, x2, y2 = sample["bbox"]
    cropped = img.crop((x1, y1, x2, y2))
    cropped.save(os.path.join(dst_dir, new_name))
    train_labels.append({"class": c, "fname": f"{c}/{new_name}"})
print("✅ Images copied successfully for training set!")

# Process Stanford Cars test images
test_labels = []
class_counts_test = defaultdict(int)
for sample in test_samples:
    c = sample["class"]
    dst_dir = os.path.join(dataset_dir, "test", "images", str(c))
    os.makedirs(dst_dir, exist_ok=True)
    class_counts_test[c] += 1
    new_name = str(class_counts_test[c]).zfill(5) + ".jpg"
    src = os.path.join(test_imgpath, sample["fname"])
    img = Image.open(src)
    x1, y1, x2, y2 = sample["bbox"]
    cropped = img.crop((x1, y1, x2, y2))
    cropped.save(os.path.join(dst_dir, new_name))
    test_labels.append({"class": c, "fname": f"{c}/{new_name}"})
print("✅ Images copied successfully for test set!")

# -------------------------------------------------------------------
# Process additional custom classes (seat_arona, seat_leon_3)
# -------------------------------------------------------------------

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # using nano model for speed, you can change it

additional_classes = [
    d for d in os.listdir(images_dir) if d in ["seat_arona", "seat_leon_3"]
]

if additional_classes:
    existing_ids = {lbl["class"] for lbl in train_labels + test_labels}
    next_class_id = max(existing_ids) + 1

    for cls_name in sorted(additional_classes):
        cls_dir = os.path.join(images_dir, cls_name)
        img_files = sorted(
            [
                f
                for f in os.listdir(cls_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        n_total = len(img_files)
        n_train = n_total // 2
        n_test = n_total - n_train

        for idx, fname in enumerate(img_files):
            src = os.path.join(cls_dir, fname)
            img = Image.open(src).convert("RGB")
            img_np = np.array(img)

            # Run YOLO detection
            results = model.predict(img_np, conf=0.25, save=False, verbose=False)[0]

            if len(results.boxes) == 0:
                print(f"⚠️ No detection for {fname}, saving full image.")
                cropped = img
            else:
                # Select the largest box
                boxes = results.boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                biggest_idx = np.argmax(areas)
                x1, y1, x2, y2 = boxes[biggest_idx].astype(int)
                cropped = img.crop((x1, y1, x2, y2))

            # Resize to max 500 pixels
            w, h = cropped.size
            max_dim = max(w, h)
            if max_dim > 500:
                scale = 500 / max_dim
                new_w = int(w * scale)
                new_h = int(h * scale)
                cropped = cropped.resize((new_w, new_h), Image.LANCZOS)

            # Determine split (first half to train, second half to test)
            if idx < n_train:
                split = "train"
                split_labels = train_labels
            else:
                split = "test"
                split_labels = test_labels

            dst_dir = os.path.join(dataset_dir, split, "images", str(next_class_id))
            os.makedirs(dst_dir, exist_ok=True)

            new_name = str(idx + 1).zfill(5) + os.path.splitext(fname)[1]
            cropped.save(os.path.join(dst_dir, new_name))

            split_labels.append(
                {"class": next_class_id, "fname": f"{next_class_id}/{new_name}"}
            )

        print(
            f"✅ Added {n_total} images for custom class '{cls_name}' (ID={next_class_id})"
        )
        next_class_id += 1
else:
    print("ℹ️  No additional custom classes found in images_dir.")

# Save labels
np.save(os.path.join(dataset_dir, "train", "labels.npy"), train_labels)
np.save(os.path.join(dataset_dir, "test", "labels.npy"), test_labels)
print("✅ Dataset prepared successfully with additional classes!")

# -------------------------------------------------------------------
# Save class ID to name and source mapping as JSON
# -------------------------------------------------------------------

meta_path = os.path.join(root, "stanford_cars", "devkit", "cars_meta.mat")
meta = scipy.io.loadmat(meta_path)
class_names = [str(name[0]) for name in meta["class_names"][0]]

class_id_to_info = {}
for i, name in enumerate(class_names):
    class_id_to_info[i] = {"name": name, "src": "StanfordCars"}

custom_class_start_id = len(class_names)
for idx, cls_name in enumerate(sorted(additional_classes)):
    class_id_to_info[custom_class_start_id + idx] = {"name": cls_name, "src": "Custom"}

# Prepare the full JSON object
class_mapping_json = {
    "total_number_of_classes": len(class_id_to_info),
    "classes": class_id_to_info,
}

mapping_path = os.path.join(dataset_dir, "meta.json")
with open(mapping_path, "w") as f:
    json.dump(class_mapping_json, f, indent=4)

print("✅ Saved class ID to name and source mapping as JSON!")
