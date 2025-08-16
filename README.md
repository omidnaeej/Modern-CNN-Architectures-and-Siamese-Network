# Modern-CNN-Architectures-and-Siamese-Network
This repository contains the implementation of **modern CNN architectures for CIFAR-10 classification** and a **Siamese Network for face recognition**, as part of Assignment 2 of the Deep Learning course.  
The goal of this project is to explore both supervised classification and similarity learning.

---

## Repository Contents
- `config/`: YAML files for training configurations and logging  
- `data/`: Data loader  
- `models/`: Model architecture definitions (CNN & Siamese)  
- `notebooks/`: Jupyter notebook for exploration and experiments  
- `scripts/`: Training, evaluation, and entry-point scripts  
- `utils/`: Metrics computation and visualization tools  
- `report.pdf`: Original report (in Persian)  

---

## Project Overview
This project demonstrates a complete deep learning workflow for **image classification** and **face verification**:

- **Data Loading & Preprocessing**  
  Custom dataloaders for CIFAR-10 and face recognition datasets with augmentation.  

- **Model Architectures**  
  - Convolutional Neural Networks for CIFAR-10 classification  
  - Siamese Network with contrastive loss for face recognition  

- **Training Pipeline**  
  Configurable training scripts with logging, checkpointing, and hyperparameter tuning.  

- **Evaluation & Metrics**  
  Accuracy, confusion matrix, and visualization of feature embeddings (e.g., t-SNE).  

- **Visualization**  
  Plotting learning curves, feature maps, and embedding spaces.  

---

## Setup
Clone the repository:
```bash
git clone https://github.com/omidnaeej/Modern-CNN-Architectures-and-Siamese-Network.git
cd Modern-CNN-Architectures-and-Siamese-Network
```

## Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the main script to train and test both models:

```bash
python -m scripts.main
```

Be aware that you can modify the configurations in the `config` folder. To try a different configuration for the classifier network, use `config.yaml`. For the Siamese network, use `config_snn.yaml`.

Note that, based on the project’s strategy, all blocks of the classifier network are identical. Therefore, `block_config` should be a list of length 1, containing only one element from the set [‘A’, ‘B’, ‘C’, ‘D’, ‘E’].

