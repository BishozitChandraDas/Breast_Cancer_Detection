# 🔬 An Explainable Deep Learning Framework for Breast Cancer Classification from Histopathological Images Using CNNs and Vision Transformers

A deep learning framework for classifying breast cancer histopathological images as **benign** or **malignant**.
This project provides a comparative study of six CNN and transformer architectures on the BreaKHis dataset, with **Grad-CAM++** explainability for clinical interpretability.

## 👤 AUTHOR
 - **Bishozit Chandra Das**   
 - Student ID: **VR544246** 
 - MSc in **Artificial Intelligence** at Università degli Studi di Verona  
 - [BishozitChandraDas](https://github.com/BishozitChandraDas)  

## 🎯 Project Overview
Breast cancer is one of the most common and deadly cancers in women, and early diagnosis is critical for survival. Manual diagnosis from histopathology slides is slow, labour-intensive, and dependent on pathologist expertise.

This project implements an **automated classifier** that distinguishes benign from malignant breast tumour tissue from histopathological images, and benchmarks several modern deep learning architectures while adding visual explainability through Grad-CAM++.

## 🧠 Problem Statement
Build an interpretable deep learning system capable of accurately classifying breast cancer histopathological images into benign and malignant classes, and compare different CNN and transformer architectures in terms of accuracy, recall, and discriminative ability — with a focus on minimising false negatives (missed malignant cases).

## 📊 Dataset Information
**BreaKHis — Breast Cancer Histopathological Image Dataset**
Source: https://www.kaggle.com/datasets/ambarish/breakhis

- Histopathological images of breast tumour tissue at multiple magnification levels
- Two classes: **Benign** and **Malignant**
- Split: **patient-wise 70 : 15 : 15** (train / validation / test) to prevent data leakage
- Test set used in this study: **1,188 images** (421 benign + 767 malignant)

## 🛠️ Methods Implemented
Six ImageNet-pretrained architectures were fine-tuned on BreaKHis:

**1. DenseNet121**
Dense connections enable strong feature reuse with fewer parameters.

**2. EfficientNetB0**
Compound scaling of depth, width, and resolution; lightweight and efficient.

**3. ResNet50**
50-layer residual network; residual connections solve the vanishing-gradient problem.

**4. Xception**
Depthwise-separable convolutions for fine-grained tissue feature extraction.

**5. MobileNetV3**
Lightweight network with squeeze-and-excitation modules and NAS; suited for low-resource deployment.

**6. Vision Transformer (ViT-B/16)**
Transformer that splits images into 16×16 patches and uses self-attention to capture global context.

## 🧩 Explainability (Grad-CAM++)
**Grad-CAM++** was applied to MobileNetV3 (selected for its lightweight, deployable design and strong accuracy/precision) to generate heatmaps highlighting the tissue regions that most influence each prediction — improving transparency and clinical trust.

## ⚙️ Experimental Setup
**Platform:** Google Colab / PyTorch
**Initialization:** ImageNet pre-trained weights → fully fine-tuned on BreaKHis

**Configuration:**
- Loss function: Cross-Entropy
- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-4 (ViT-B/16, Xception, MobileNetV3)
- Batch size: 32
- Epochs: 10
- Image size: 224 × 224
- Augmentation: random horizontal flip + rotation
- Normalization: ImageNet mean / std
- Model selection: best validation checkpoint saved

## 📈 Performance Results

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | AUC (%) |
|---|---|---|---|---|---|
| ViT-B/16 | 81.57 | 82.70 | **90.35** | **86.36** | 88.39 |
| MobileNetV3 | **81.82** | **84.65** | 87.74 | 86.17 | 85.00 |
| ResNet50 | 81.48 | 84.23 | 87.74 | 85.95 | **88.55** |
| Xception | 79.38 | 81.07 | 88.79 | 84.75 | 84.46 |
| EfficientNetB0 | 78.28 | 83.01 | 83.44 | 83.22 | 85.03 |
| DenseNet121 | 76.18 | 80.56 | 83.18 | 81.85 | 84.04 |

Metrics reported on the BreaKHis test set. **Bold** = best value per column.

## 🔍 Key Findings
- No single model wins everything — three models lead on different metrics:
  - **ViT-B/16** — best recall and F1 (catches the most malignant cases, fewest missed)
  - **ResNet50** — best AUC (most robust across decision thresholds)
  - **MobileNetV3** — best accuracy and precision (top balance, lightweight)
- All six architectures reliably separate benign from malignant tissue.
- Both CNNs and transformer-based models are effective for this task.
- Grad-CAM++ adds the transparency that clinical adoption requires.
- CPU-only training would be slow; GPU acceleration is recommended.

## 📋 Requirements

**Environment**
- Platform: Google Colab (recommended) or local machine
- Python: 3.10+
- Hardware: GPU recommended (CPU possible but slow)

**Dependencies**
Install the required libraries before running the notebook:

```
pip install torch
pip install torchvision
pip install timm
pip install grad-cam
pip install scikit-learn
pip install matplotlib
pip install numpy
pip install pandas
pip install Pillow
```

## 🔧 Setup

**1. Google Colab (Recommended)**
- Open Google Colab
- Upload the project notebook (`breast_cancer_detection.ipynb`)
- Install the required dependencies
- Download the BreaKHis dataset from Kaggle and point the notebook to the image folder
- Run all cells sequentially

**2. Local Environment**
- Clone the repository
- Create and activate a Python virtual environment
- Install the dependencies listed above
- Download the BreaKHis dataset and set the dataset path
- Run the notebook or script

## ▶️ How to Run
1. Clone the repository
2. Open the notebook in Google Colab (or locally)
3. Install the required dependencies
4. Set the BreaKHis dataset path
5. Run all cells sequentially to train, evaluate, and generate Grad-CAM++ visualizations

## 📌 Limitations
- Class imbalance in the dataset (more malignant than benign samples)
- Limited hyperparameter tuning (10 epochs, fixed learning rate)
- Training time can be long without a GPU
- Single magnification handling — magnification-specific analysis not separated

## 🎓 Academic Context
This project was developed as part of the **Machine Learning and Deep Learning (Deep Learning)** coursework for the MSc in Artificial Intelligence at the University of Verona, focusing on:

- Transfer learning with CNN and transformer architectures
- Performance comparison across model families
- Explainable AI for medical image classification
- Trade-offs between accuracy, recall, and computational cost

## 📄 Notes
- Intended for educational and research purposes
- Not optimized or validated for clinical/production use
