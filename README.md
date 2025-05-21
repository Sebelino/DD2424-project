# MeowWoofNet

<p align="center">
<img src="https://github.com/user-attachments/assets/e0c0d99f-fe50-42b1-9ff7-d2361992d9c3" alt="project icon" width="256" />
</p>
<p align="center">
  <em>A Hybrid Semi-Supervised and Masked Fine-Tuning Framework for Pet Recognition</em>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python Version](https://img.shields.io/badge/python-3.8%2B-green.svg)]()  
[![PyTorch Version](https://img.shields.io/badge/pytorch-%3E%3D1.9-orange.svg)]()

---

## Project Overview

MeowWoofNet is a research-oriented framework developed as a group project for the DD2424 “Deep Learning in Data Science” course at KTH. It targets robust pet breed recognition on the Oxford-IIIT Pet Dataset by combining:

- **Semi-Supervised Learning**: Leverages unlabeled data via consistency regularization.  
- **Masked Fine-Tuning**: Applies input masking during fine-tuning to encourage the model to learn more discriminative features.  

At its core, MeowWoofNet is built on a ResNet-50 backbone and PyTorch, and provides end-to-end pipelines for data handling, model training, evaluation, visualization, and reproducible experiments.

---

## Features

- **Download & Preprocess** the Oxford-IIIT Pet Dataset with one script.  
- **Flexible Data Augmentation** (random crops, flips, color jitter, cutout, etc.).  
- **Semi-Supervised Module** for leveraging unlabeled splits.  
- **Masked Fine-Tuning** to improve generalization on small datasets.  
- **Deterministic & Reproducible** setups with seed control and caching.  
- **Automated Training & Evaluation** via `run.py` with configurable hyperparameters.  
- **Visualization Tools** for learning curves, confusion matrices, and augmentation examples.  
- **Jupyter Notebooks** for guided tutorials (E1, E2, E3).
