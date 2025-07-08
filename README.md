# Security and robustness of prototype-based explanation methods, especially against adversarial attacks
Master Final Thesis - Research into the security and robustness of prototype-based explanation methods within XAI.

This repository contains the code and resources for the Master's Final Thesis on **"Security and Robustness of Prototype-Based Explanation Methods in Explainable Artificial Intelligence (XAI)"**.


Autor: Silvia Arenales Muñoz
## Introduction

Explainable Artificial Intelligence (XAI) has become increasingly important for developing transparent and trustworthy machine learning systems. Among the different XAI techniques, **prototype-based explanation methods** are notable for their ability to provide intuitive, human-understandable interpretations by referencing representative examples from the training set.

However, the **robustness** and **security** of these explanations under adversarial settings remain underexplored. Adversarial attacks can subtly manipulate input data to deceive both models and their associated explanations. This project investigates how prototype-based explanations behave under such attacks and explores strategies to **improve their resilience**.

Through theoretical insights and empirical evaluations, the project contributes to a better understanding of how XAI methods can be **hardened** against adversarial manipulations.

![alt text](https://github.com/sarenales/PrototypesVulnerabilities/blob/main/images/GoodFellow2014.png "GoodFellow 2014 paper")

---

##  Installation

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

## How to Run
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

## Attacks implemented
The project includes several adversarial attack methods:
- DeepFool
- Projected Gradient Descent (PGD / APGD)
- EAD (Elastic-net Attack on DNNs)
- SparseFool
- Pixle Attack

---
## Goals of the Project

Evaluate how prototype-based explanations degrade under adversarial conditions.

Compare attack methods in terms of their ability to alter both model predictions and explanations.

Propose methods (e.g., fine-tuning, robust training) to improve explanation reliability.

---

## Future Directions

- Scalable evaluation frameworks for prototype-based models under diverse adversarial settings
- New **architectures or loss functions** tailored for explanation robustness
- Further exploration of **interpretability as an attack vector**, rather than just a defense mechanism

---
## Implications

Our results underscore a **fundamental paradox**:

> **Transparency enables trust, but also opens doors for adversarial exploitation.**

This has implications for:
- **Cybersecurity protocols**, particularly regarding zero-day adversarial vulnerabilities
- The design of **robust and interpretable AI systems**
- The development of **explanation-aware defenses** that preserve both accuracy and human trust


----
This project is for academic and research purposes. Please contact the author for reuse or extension in other contexts.
