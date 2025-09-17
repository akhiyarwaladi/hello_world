#!/bin/bash
# RT-DETR Classification Pipeline Script
# Waits for RT-DETR crop generation to complete, then runs classification training

echo "🔄 Waiting for RT-DETR crop generation to complete..."

# Wait for crop generation to complete by checking if the directory exists and has content
while [ ! -d "data/crops_from_rtdetr_detection/yolo_classification" ] || [ -z "$(ls -A data/crops_from_rtdetr_detection/yolo_classification 2>/dev/null)" ]; do
    echo "⏳ Waiting for RT-DETR crops... (checking every 30 seconds)"
    sleep 30
done

echo "✅ RT-DETR crops ready! Starting classification training..."

# Start all RT-DETR classification training in parallel
echo "🚀 Starting RT-DETR → Classification Training Pipeline"

# YOLO Classification Models
NNPACK_DISABLE=1 python scripts/11_train_classification_crops.py \
    --data "data/crops_from_rtdetr_detection/yolo_classification" \
    --model "yolov8n-cls.pt" \
    --epochs 10 \
    --batch 4 \
    --device cpu \
    --name "rtdetr_det_to_yolo8_cls" &

NNPACK_DISABLE=1 python scripts/11_train_classification_crops.py \
    --data "data/crops_from_rtdetr_detection/yolo_classification" \
    --model "yolo11n-cls.pt" \
    --epochs 10 \
    --batch 4 \
    --device cpu \
    --name "rtdetr_det_to_yolo11_cls" &

# PyTorch Classification Models
NNPACK_DISABLE=1 python scripts/11b_train_pytorch_classification.py \
    --data "data/crops_from_rtdetr_detection/yolo_classification" \
    --model "resnet18" \
    --epochs 10 \
    --batch 8 \
    --device cpu \
    --name "rtdetr_det_to_resnet18" &

NNPACK_DISABLE=1 python scripts/11b_train_pytorch_classification.py \
    --data "data/crops_from_rtdetr_detection/yolo_classification" \
    --model "efficientnet_b0" \
    --epochs 10 \
    --batch 8 \
    --device cpu \
    --name "rtdetr_det_to_efficientnet" &

echo "🎯 Started 4 RT-DETR classification training processes in background"
echo "✅ RT-DETR pipeline initiated successfully!"

# Wait for all background processes to complete
wait

echo "🎉 All RT-DETR classification training completed!"