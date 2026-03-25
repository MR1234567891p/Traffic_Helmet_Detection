# Helmet Detection using YOLOv8

## Overview

This project implements a **helmet detection system** using the YOLOv8 object detection model. The goal is to identify whether motorbike riders are wearing helmets or not in real-world traffic images, contributing to improved road safety and automated traffic monitoring systems.

---

## Objectives

* Detect motorbike riders in images
* Classify riders as **helmet** or **no helmet**
* Build a reliable object detection model using YOLOv8
* Evaluate performance using standard metrics

---

## Dataset

* **Source:** Kaggle – Motorbike Helmet Detection Dataset
* **Total Images:** 764
* **Annotated Images:** 747
* **Classes:**

  * Helmet
  * No Helmet
* **Total Objects:** 1420

The dataset contains real-world traffic scenarios, making it suitable for practical applications.

---

## Project Pipeline

### 1️⃣ Data Preprocessing

* Converted XML annotations to YOLO format
* Normalized class labels
* Split dataset into training and testing sets (80–20)
* Generated `data.yaml` file for YOLOv8

### 2️⃣ Model Training

* Model: YOLOv8 (Nano variant)
* Epochs: 50
* Image Size: 416
* Batch Size: 2
* Device: CPU

### 3️⃣ Evaluation Metrics

* mAP@50
* mAP@50–95
* Precision
* Recall

---

## Results

| Metric    | Value  |
| --------- | ------ |
| mAP@50    | 0.8079 |
| mAP@50–95 | 0.4828 |
| Precision | 0.7881 |
| Recall    | 0.7697 |

The model demonstrates strong performance in detecting helmet usage in real-world conditions.

---

## Features

* End-to-end pipeline (data preparation → training → testing)
* Automatic annotation conversion to YOLO format
* Works on real-world traffic images
* Lightweight model suitable for deployment

---

## Tech Stack

* Python
* YOLOv8 (Ultralytics)
* OpenCV
* XML Parsing
* Scikit-learn

---

## 📂 Project Structure

```
Traffic_Helmet_Detection/
│
├── annotations/          # XML annotation files
├── images/               # Dataset images
├── train.py              # Training script
├── val.py                # Validation script
├── test.py               # Testing script
├── prepare_dataset.py    # Dataset preprocessing
├── dataset_check.py      # Dataset analysis
└── README.md
```

---

## 🚀 How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/MR1234567891p/Traffic_Helmet_Detection.git
cd Traffic_Helmet_Detection
```

### 2️⃣ Install dependencies

```bash
pip install ultralytics opencv-python scikit-learn
```

### 3️⃣ Prepare dataset

```bash
python prepare_dataset.py
```

### 4️⃣ Train model

```bash
python train.py
```

### 5️⃣ Evaluate model

```bash
python val.py
```

### 6️⃣ Test model

```bash
python test.py
```

---

## Model Weights

Trained model weights (`best.pt`) are not included due to size limitations.
The model can be reproduced using the provided training script.

---

## Applications

* Traffic monitoring systems
* Automated fine detection
* Smart city solutions
* Road safety enforcement

---

## Future Work

* Train on larger datasets
* Use GPU for faster training
* Extend to real-time video detection
* Improve detection accuracy

