# Security and robustness of prototype-based explanation methods, especially against adversarial attacks
Master Final Thesis - Research into the security and robustness of prototype-based explanation methods within XAI.

This repository contains the code and resources for the Master's Final Thesis on **"Security and Robustness of Prototype-Based Explanation Methods in Explainable Artificial Intelligence (XAI)"**.


Autor: Silvia Arenales Muñoz
## Introduction

Explainable Artificial Intelligence (XAI) has become increasingly important for developing transparent and trustworthy machine learning systems. Among the different XAI techniques, **prototype-based explanation methods** are notable for their ability to provide intuitive, human-understandable interpretations by referencing representative examples from the training set.

However, the **robustness** and **security** of these explanations under adversarial settings remain underexplored. Adversarial attacks can subtly manipulate input data to deceive both models and their associated explanations. This project investigates how prototype-based explanations behave under such attacks and explores strategies to **improve their resilience**.

Through theoretical insights and empirical evaluations, the project contributes to a better understanding of how XAI methods can be **hardened** against adversarial manipulations.


## 🗂 Project Structure

📁 data/
└── MNIST/ # MNIST dataset
📁 saved_model/
└── mnist_model/ # Pretrained models
📄 Adversarial_Attacks_Testing.ipynb # Main notebook for testing attacks
📄 PrototypeDL_MNIST_Visualization.ipynb # Visualization of prototypes
📄 View_FineTune.ipynb # Visualizing fine-tuning effects
📄 Attack.py # Base class for attack methods
📄 adversarial_attacks.py # Wrapper for multiple attack strategies
📄 apgd.py # APGD attack implementation
📄 deepfool.py # DeepFool attack implementation
📄 eaden.py # EADEN (Elastic-Net) attack
📄 eadl1.py # EAD L1-variant
📄 pixle.py # Pixle attack
📄 sparsefool.py # SparseFool attack
📄 finetune_model.py # Fine-tuning the prototype model
📄 finetune_attacks.py # Adversarial fine-tuning
📄 model_testing.py # Testing utilities
📄 autoencoder_helpers.py # Helper functions for autoencoder
📄 data_loader.py # Data loading utilities
📄 data_preprocessing.py # Data preprocessing steps
📄 loss_functions.py # Custom loss functions
📄 modules.py # Model and prototype modules
📄 train_mnist.py # Training base MNIST model
📄 train_mnist_adv.py # Training model with adversarial robustness
📄 ftb30preconst.png # Example image from results
📄 test.drawio # Architecture diagram (editable with draw.io)
📄 requirements.txt # Python dependencies
📄 .gitignore # Git ignore list
📄 README.md # Project documentation



---

## ⚙️ Installation

1. **Clone the repository**
   ```
   git clone https://github.com/yourusername/prototype-xai-robustness.git
   cd prototype-xai-robustness
   ```
2. **Create a virtual enviroment (optionak but recommended)**
  ```
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```
3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```
  
---

## 🚀 How to Run
- **Train a basic MNIST classifer**
  ```
  python train_mnist.py
  ```
- **Train a robust model**
  ```
  python train_mnist_adv.py
  ```
- **Run adversarial attacks**
  Open and execute the notebook ```Adversarial_Attacks_Testing.ipynb```.
- **Visualize prototypes and explanations**
  Use ```PrototypeDL_MNIST_Visualization.ipynb```.

---

## 🧪 Attacks implemented
The project includes several adversarial attack methods:
- DeepFool
- Projected Gradient Descent (PGD / APGD)
- EAD (Elastic-net Attack on DNNs)
- SparseFool
- Pixle Attack

---
## 🧠 Goals of the Project

Evaluate how prototype-based explanations degrade under adversarial conditions.

Compare attack methods in terms of their ability to alter both model predictions and explanations.

Propose methods (e.g., fine-tuning, robust training) to improve explanation reliability.


----
This project is for academic and research purposes. Please contact the author for reuse or extension in other contexts.
