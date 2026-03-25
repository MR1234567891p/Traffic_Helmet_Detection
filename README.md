
##  Helmet Detection using YOLOv8

This project implements a **helmet detection system** using the Motorbike Helmet Detection dataset and the YOLOv8 object detection model. The goal is to identify riders **with and without helmets** in real-world traffic images, contributing to road safety applications.

### Dataset

* **Total images:** 764
* **Annotated images:** 747
* **Classes:** `helmet`, `no helmet`
* **Total objects:** 1420
* Balanced class distribution with real-world traffic scenarios

### Project Pipeline

1. **Data Preprocessing**

   * Converted XML annotations to YOLO format
   * Normalized class labels
   * Split dataset into train/test sets
   * Generated `data.yaml` for training

2. **Model Training**

   * Model: YOLOv8 (nano variant)
   * Image size: 416
   * Epochs: 50
   * Trained on CPU

3. **Evaluation**

   * mAP@50: **0.8079**
   * mAP@50–95: **0.4828**
   * Precision: **0.7881**
   * Recall: **0.7697**

### Results

The model demonstrates strong performance in detecting helmet usage, making it suitable for **traffic monitoring and safety enforcement systems**.

### Features

* End-to-end pipeline (data prep → training → testing)
* Automated dataset conversion for YOLOv8
* Reliable detection in real-world scenarios

### Tech Stack

* Python
* Ultralytics YOLOv8
* OpenCV, XML parsing
* Scikit-learn

### Code

Includes scripts for:

* Dataset preparation
* Model training
* Validation
* Testing 


