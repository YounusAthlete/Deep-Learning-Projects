# 🌟 Skin Lesion Classification Using Deep Learning

## 📌 Introduction
This project focuses on **classifying skin lesions** using the **HAM10000 dataset**. It leverages advanced deep learning models, including:
- **Vision Transformers (ViT)**
- **Hybrid ViT-VGG Model**

The goal is to achieve accurate classification of skin lesions into seven categories.

---

## 📂 Dataset: HAM10000
The **HAM10000** dataset contains **10,015 dermatoscopic images** of skin lesions categorized into seven classes:
1. **Melanoma (mel)**
2. **Basal Cell Carcinoma (bcc)**
3. **Actinic Keratosis (akiec)**
4. **Benign Keratosis-like Lesion (bkl)**
5. **Dermatofibroma (df)**
6. **Vascular Lesion (vasc)**
7. **Nevi (nv)**

### 🔍 Metadata Includes:
- **Lesion ID**
- **Image ID**
- **Diagnosis Type**
- **Age, Sex, and Localization**

*Images are consolidated from two folders into a single directory for processing.*

---

## 🛠 Methodology
This project follows a standard deep learning workflow:

1. **Data Preparation**
   - Download, extract, and preprocess the dataset.
   - Resize and normalize images to fit model input requirements.

2. **Model Selection**
   - **Vision Transformer (ViT)**
   - **Hybrid ViT-VGG Model**

3. **Training**
   - **Optimizer**: AdamW with a learning rate scheduler.
   - **Data Augmentation**: Applied to minority classes to address class imbalance.

4. **Evaluation**
   - Metrics: **Accuracy** and **Classification Report**
   - Hybrid model is also tested on individual images.

---

## 🧠 Model Architectures

### 1. Vision Transformer (ViT)
- **Pretrained Model**: `google/vit-base-patch16-224-in21k`
- **Fine-tuning**: Replaced final classification layer for 7 classes.
- **Training**:
  - **Epochs**: 7
  - **Learning Rate**: 1e-4

### 2. Hybrid ViT-VGG Model
- **ViT Component**: Feature extraction using the pretrained ViT model.
- **VGG16 Component**: Extracts additional features.
- **Combination**: Concatenates outputs from ViT and VGG16.
- **Training**:
  - **Epochs**: 5
  - **Learning Rate**: 1e-4

---

## ⚙️ Training Process
- **Data Augmentation**: Techniques like random horizontal flipping, rotation, and color jittering.
- **Loss Function**: Cross-Entropy Loss.
- **Optimizer**: AdamW with a StepLR learning rate scheduler.
- **Dataset Split**: 80% Training and 20% Validation.

---

## 📊 Results

### Vision Transformer (ViT)
- **Validation Accuracy**: **89.17%**
- **Classification Report**:
  - High precision and recall for classes with more samples (e.g., Nevi).
  - Lower performance for minority classes (e.g., Dermatofibroma, Vascular Lesion).

### Hybrid ViT-VGG Model
- **Validation Accuracy**: **85.52%**
- **Classification Report**:
  - Competitive performance with high accuracy on Nevi.
  - Struggles with minority classes, similar to ViT.

---

## 💡 Discussion
- **Model Performance**:
  - High accuracy on majority class (Nevi).
  - Lower performance on minority classes due to class imbalance.
- **Data Augmentation**:
  - Effective but requires more advanced techniques (e.g., GANs).
- **Hybrid Model**:
  - Promising but needs further architectural tuning.

---

## ✅ Conclusion
- Successfully applied **Vision Transformer** and **Hybrid ViT-VGG** models for skin lesion classification.
- Achieved **high accuracy on majority class** but requires improvement for minority classes.
- **Future Directions**:
  - Advanced data augmentation techniques.
  - Address class imbalance.
  - Experiment with other hybrid architectures.

---

## 🚀 Future Work
1. **Class Imbalance**:
   - Implement techniques like **SMOTE** or **GANs** for synthetic data generation.

2. **Model Tuning**:
   - Experiment with hyperparameters and architecture adjustments.

3. **Transfer Learning**:
   - Explore other pretrained models like **ResNet** and **EfficientNet**.

4. **Deployment**:
   - Develop a user-friendly interface for real-time skin lesion classification.

---

## 🔗 References
- **[HAM10000 Dataset on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)**
- **[Vision Transformer (ViT) - Hugging Face](https://huggingface.co/google/vit-base-patch16-224-in21k)**
- **[VGG16 - Keras Applications](https://keras.io/api/applications/#vgg16)**

---

## 📸 Sample Predictions
| Image | Predicted Class | Confidence |
|-------|-----------------|------------|
| ![Sample 1](./images/sample1.png) | Nevi (nv) | 93% |
| ![Sample 2](./images/sample2.png) | Melanoma (mel) | 87% |
| ![Sample 3](./images/sample3.png) | Basal Cell Carcinoma (bcc) | 85% |

---

## 📁 Directory Structure
