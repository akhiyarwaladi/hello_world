# The code implements single-class object detection (identifying only parasite presence/absence) while preserving all 
# original bounding box annotations

import os
import shutil
import yaml
import cv2
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from IPython.display import Image, display
import random

# Checking GPU Availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 1. Preparing the data structure ==============================================

# Source paths (Kaggle input)
input_base = "/kaggle/input/yolo-formatted-mp-idb-malaria-dataset/MP-IDB-YOLO"
all_images_dir = os.path.join(input_base, "images")
all_labels_dir = os.path.join(input_base, "labels")

# Working directories (Kaggle working)
working_base = "/kaggle/working/MP-IDB-YOLO"
os.makedirs(working_base, exist_ok=True)

# Create a folder structure
train_img_dir = os.path.join(working_base, "images/train")
val_img_dir = os.path.join(working_base, "images/val")
train_label_dir = os.path.join(working_base, "labels/train")
val_label_dir = os.path.join(working_base, "labels/val")

for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Data separation
image_files = [f for f in os.listdir(all_images_dir) if f.lower().endswith(('.jpg', '.png'))]
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

# Copying files
def copy_files(files, src_img_dir, src_label_dir, dst_img_dir, dst_label_dir):
    for img_file in files:
        # Copying the image
        shutil.copy(
            os.path.join(src_img_dir, img_file),
            os.path.join(dst_img_dir, img_file))
        
        # Copy the corresponding label
        label_file = os.path.splitext(img_file)[0] + '.txt'
        shutil.copy(
            os.path.join(src_label_dir, label_file),
            os.path.join(dst_label_dir, label_file))

copy_files(train_files, all_images_dir, all_labels_dir, train_img_dir, train_label_dir)
copy_files(val_files, all_images_dir, all_labels_dir, val_img_dir, val_label_dir)

# Convert all classes to single "parasite" class (0)
def convert_to_single_class(labels_dir):
    for label_file in os.listdir(labels_dir):
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            lines = [line for line in f if line.strip()]  # Keep all bboxes
        
        # Convert all class IDs to 0
        new_lines = []
        for line in lines:
            parts = line.split()
            parts[0] = '0'  # Change class ID to 0
            new_lines.append(" ".join(parts) + "\n")
        
        with open(os.path.join(labels_dir, label_file), 'w') as f:
            f.writelines(new_lines)

# Apply conversion to all labels
convert_to_single_class(train_label_dir)
convert_to_single_class(val_label_dir)

# Create data.yaml with one class
yaml_content = f"""
train: {train_img_dir}/
val: {val_img_dir}/
nc: 1  # Only detect presence/absence
names: ['parasite']
"""

DATA_YAML_PATH = os.path.join(working_base, "data.yaml")
with open(DATA_YAML_PATH, "w") as f:
    f.write(yaml_content)

print(f"\nDataset prepared at: {working_base}")
print(f"Train images: {len(train_files)}")
print(f"Val images: {len(val_files)}")
print("Number of classes: 1 (parasite)")

# 2. Checking the data structure ===============================================

def check_dataset_structure(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    
    print("\nDataset info:")
    print(f"Classes: {data['names']}")
    print(f"Number of classes: {data['nc']}")
    
    # Checking the number of files
    train_count = len(os.listdir(data['train']))
    val_count = len(os.listdir(data['val']))
    
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")
    
    # Checking images and labels compliance
    train_labels = os.path.dirname(data['train']).replace('images', 'labels')
    train_label_count = len(os.listdir(train_labels))
    print(f"Training labels: {train_label_count} (should match images)")
    
    return data

dataset_info = check_dataset_structure(DATA_YAML_PATH)

# 3. Class distribution analysis ===========================================

def analyze_class_distribution(labels_dir):
    class_counts = {0: 0}  # Only parasite class
    
    for label_file in os.listdir(labels_dir):
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1
    
    print("\nClass distribution in training set:")
    print(f"parasite: {class_counts[0]} instances")
    
    # Distribution visualization
    plt.figure(figsize=(6, 4))
    plt.bar(['parasite'], [class_counts[0]])
    plt.title("Class Distribution in Training Set")
    plt.ylabel("Number of instances")
    plt.tight_layout()
    plt.show()
    
    return class_counts

class_distribution = analyze_class_distribution(train_label_dir)

# 4. Model training ========================================================

MODEL_NAME = "yolo11n"  # Using YOLO11n
EPOCHS = 100
IMGSZ = 640
BATCH_SIZE = 16
CONF_THRES = 0.3

# Loading the model
model = YOLO(f"{MODEL_NAME}.pt").to(device)

# Learning function
def train_model():
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        name=f'{MODEL_NAME}_malaria_single_class',
        patience=15,
        save=True,
        save_period=10,
        device=device,
        workers=4,
        exist_ok=True,
        optimizer='AdamW',
        lr0=0.001,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=45,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.2
    )
    return results

print("\nStarting training...")
train_results = train_model()

# 5. Model Validation ======================================================

def validate_model():
    metrics = model.val(
        data=DATA_YAML_PATH,
        batch=BATCH_SIZE,
        imgsz=IMGSZ,
        conf=0.25,
        iou=0.45,
        device=device,
        split='val'
    )
    
    print("\nValidation metrics:")
    print(f"mAP50-95: {metrics.box.map:.4f}")  
    print(f"mAP50: {metrics.box.map50:.4f}")    
    print(f"mAP75: {metrics.box.map75:.4f}")    
    
    # Handle array-based metrics (precision and recall)
    if hasattr(metrics.box.p, '__iter__'):  # Check if it's an array
        print("\nPer-class Precision:", [f"{x:.4f}" for x in metrics.box.p])
        print("Per-class Recall:", [f"{x:.4f}" for x in metrics.box.r])
    else:  # Fallback for single values
        print(f"Precision: {metrics.box.p:.4f}")
        print(f"Recall: {metrics.box.r:.4f}")
    
    return metrics

val_metrics = validate_model()

# 6. Visualization of results ==============================================

def predict_and_visualize(image_path, conf_thres=CONF_THRES):
    """Prediction and visualization of results"""
    if not os.path.exists(image_path):
        print(f"Error: Image not found - {image_path}")
        return None
    
    results = model.predict(
        source=image_path,
        conf=conf_thres,
        imgsz=IMGSZ,
        device=device
    )
    
    for r in results:
        im_array = r.plot()
        im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(im_rgb)
        plt.axis('off')
        plt.show()
        
        print(f"\nDetections in {os.path.basename(image_path)}:")
        for box in r.boxes:
            cls_id = int(box.cls)
            cls_name = model.names[cls_id]
            conf = box.conf.item()
            print(f"- {cls_name}: confidence {conf:.2f}")
    
    return results

# Example of prediction on a random image from the validation set
val_images = os.listdir(val_img_dir)
if val_images:
    sample_image = os.path.join(val_img_dir, random.choice(val_images))
    predict_and_visualize(sample_image)
else:
    print("No validation images found for demonstration")

# 7. Export model ========================================================

print("\nExporting model to ONNX format...")
export_success = model.export(format="onnx", imgsz=IMGSZ)
if export_success:
    print("Model successfully exported to ONNX format")
else:
    print("Model export failed")

print("\nTraining pipeline completed!")
