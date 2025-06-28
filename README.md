# 🐔 Poultry Disease Detection using Deep Learning

A deep learning project using transfer learning (VGG16) to classify poultry feces images into four classes:  
`Coccidiosis`, `Healthy`, `Newcastle`, and `Salmonella`.


---

## 📁 Table of Contents

- [About the Project](#about-the-project)
- [Demo](#demo)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [How to Run](#how-to-run)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Contributors](#contributors)
- [License](#license)

---

## 📌 About the Project

This AI model helps detect poultry diseases based on feces images, aiding in faster and smarter poultry health management.

> **Goal**: Enhance disease diagnosis efficiency and reduce poultry mortality.

---

## 🎬 Demo

(https://drive.google.com/file/d/1lX2l2eT4d6xvfpMudZ6RffRU_eQYZgCP/view?usp=drive_link)

---

## 📊 Dataset

- Source: Custom collected / Kaggle / [Your Data Source]
- Total Images: `XXXX`
- Classes:  
  - 🟥 Coccidiosis  
  - 🟩 Healthy  
  - 🟦 Newcastle  
  - 🟨 Salmonella  

---

## 🧠 Model Architecture

- Pretrained Model: `VGG16`
- Fine-tuned layers added:
  - `GlobalAveragePooling2D`
  - `Dense(128, activation='relu')`
  - `Dropout`
  - `Dense(4, activation='softmax')`

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
