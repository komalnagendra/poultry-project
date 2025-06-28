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

- Source: Custom collected / Kaggle / chandrashekarnatesh/poultry-diseases
- Total Images: 2000
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

---
# Setup & Run
  1.Clone the repository:
   git clone https://github.com/komalnagendra/poultry-project
   cd poultry-disease-prediction
  2.Save the files as below:
    poultry-project
    ├── static/
    │ └── uploads/  
    | └── css & js files
    ├── templates/
    │   └── html files     
    ├── healthy vs rotten.h5             
    ├── app.py               
    └── requirements.txt              
  3.Run the application:
    python app.py


