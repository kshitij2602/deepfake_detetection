# deepfake_detetection
This repository contains the implementation of the master’s capstone project “Deepfake Detection Using Deep Learning Models and Interactive Dashboard Deployment.” It focuses on detecting AI-generated faces using the Kaggle “140K Real and Fake Faces” dataset, which combines real FFHQ images with StyleGAN-generated fake faces at 256×256 resolution.
trained model output files link:-
kaggle kernels output skshitij/deepfake-detection -p /path/to/dest



## Kaggle notebook link for reference and model outputs 
https://www.kaggle.com/code/skshitij/deepfake-detection/notebook
# Deepfake Detection Using Deep Learning Models

This repository contains the implementation of the master’s capstone project **“Deepfake Detection Using Deep Learning Models and Interactive Dashboard Deployment.”**[file:1] The project focuses on detecting AI-generated faces using the Kaggle **140K Real and Fake Faces** dataset (70k real FFHQ images and 70k StyleGAN-generated fakes at 256×256 resolution).[file:1]

## Project Overview

The goal is to build and compare deep learning models that classify facial images as **real** or **fake** under realistic GPU and dataset constraints.[file:1] The project also includes an interactive Streamlit dashboard for model comparison and image/video inference.[file:1]

### Implemented Models

- **Baseline CNN** – Custom 3-block CNN trained from scratch as a domain-specific detector.[file:1]  
- **ResNet50** – ImageNet-pretrained backbone with custom head and two-stage fine-tuning.[file:1]  
- **EfficientNetB0** – ImageNet-pretrained backbone with limited fine-tuning due to GPU time constraints.[file:1]

All models are trained on 100,000 images, validated on 20,000, and tested on 20,000.[file:1]

### Key Results (Test Set)

| Model          | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---------------|----------|-----------|--------|----------|---------|
| Baseline CNN   | 0.88     | 0.91      | 0.88   | 0.90     | 0.97    |
| ResNet50       | 0.73     | 0.73      | 0.73   | 0.73     | 0.80    |
| EfficientNetB0 | 0.55     | 0.46      | 0.50   | 0.33     | 0.60    |[file:1]

The custom CNN outperforms the ImageNet-pretrained models, highlighting the value of domain-specific architectures for forensic tasks.[file:1]

## Dataset

- **Source:** Kaggle – *140K Real and Fake Faces*  
- **Classes:** 70,000 real FFHQ images, 70,000 StyleGAN fakes.[file:1]  
- **Splits:** 50k/10k/10k real and 50k/10k/10k fake for train/val/test.[file:1]  
- **Preprocessing:** Resizing (224×224 or 256×256), normalization to \([0,1]\), real-time augmentation (rotation, shifts, flips, zoom).[file:1]

## Tech Stack

- **Frameworks:** TensorFlow 2.x, Keras, Streamlit.[file:1]  
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, OpenCV, Pillow.[file:1]  
- **Environment:** Kaggle Notebooks with NVIDIA Tesla P100 GPU (16GB VRAM).[file:1]

## Streamlit Dashboard

The `dashboard.py` app provides:[file:1]

- **Model Comparison** – Metrics table, ROC curves, and bar charts across all three models.[file:1]  
- **Image Demo** – Upload an image (JPG/PNG), select a model, and get REAL/FAKE prediction with confidence and warnings.[file:1]  
- **Video Demo** – Upload a video, run frame-wise detection, aggregate predictions, and visualize suspicious segments over time.[file:1]

### Running the Dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard.py
