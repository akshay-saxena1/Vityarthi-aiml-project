# 🍔 Pro Food Analyzer & Nutrition Estimator

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)

> An end-to-end Machine Learning application that uses a custom-trained Vision Transformer (ViT) to instantly classify 101 different types of food from images and dynamically generate macronutrient profiles.

---

## ✨ Key Features-

* **🧠 Custom Fine-Tuned ViT:** Built on `google/vit-base-patch16-224-in21k` and fine-tuned on the massive 5GB **Food-101** dataset.
* **⚡ Hardware-Optimized Pipeline:** Engineered to train efficiently on a 6GB VRAM GPU (RTX 4050) using Mixed Precision (FP16), Gradient Accumulation, and Multi-core CPU Dataloading.
* **📊 Interactive Dashboard:** A sleek, reactive UI built with Streamlit that renders custom bar charts and nutritional metrics instantly.
* **🛡️ Algorithmic Fallback Estimator:** A robust fail-safe that algorithmically generates realistic macro-profiles for unrecognized items, ensuring the application never crashes and the UI always renders.

---

## 📈 Model Performance & Metrics

The model underwent a highly optimized 3-epoch master training run utilizing a Cosine Annealing learning rate scheduler, achieving professional-grade accuracy on unseen validation data:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | `88.87%` |
| **F1 Score (Macro)** | `88.83%` |
| **Precision (Macro)** | `88.86%` |
| **Recall (Macro)** | `88.87%` |

---

## 🚀 Local Setup & Installation


**1. Clone the repository**
```bash
git clone https://github.com/akshay-saxena1/Vityarthi-aiml-project.git
cd Food_AI_Project
```

**2. Install Dependencies**
```bash
pip install torch torchvision torchaudio transformers datasets evaluate streamlit pandas seaborn matplotlib
```

**3. Train the Master Model**
Run the highly-optimized training script to download the Food-101 dataset and bake the model locally. (Estimated time on RTX 4050: ~45 minutes).
```bash
python train_master.py
```

**4. Launch the Web Application**
Once the best_model is saved, boot up the interactive Streamlit interface:

```bash
streamlit run food_app.py
```

---

👨‍💻 Author

Akshay Saxena 

B.Tech Computer Science Engineering (Core)

VIT Bhopal University

**Note: Developed as a "Bring Your Own Project" submission demonstrating end-to-end AI application architecture, from raw data processing and hardware-constrained model training to final UI/UX deployment.*
