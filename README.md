# **CNN_Comparison_on_Static_Hand_Gesture**

This repository contains the codebase, experimental notebooks, and assets for the research project titled **"A Comparative Study on Performance of Pretrained ImageNet Models for Character Level Static Hand Gesture Recognition."** The study focuses on evaluating the performance of 15 pretrained convolutional neural network (CNN) architectures for static hand gesture recognition using the **Sign Language Gesture Images Dataset**.

---

## **Repository Structure**

```
├── Notebooks/
│   ├── character-sign-language-detection-DenseNet121.ipynb
│   ├── character-sign-language-detection-DenseNet169.ipynb
│   ├── character-sign-language-detection-DenseNet201.ipynb
│   ├── character-sign-language-detection-ConvNeXt.ipynb
│   ├── character-sign-language-detection-ConvNeXtBase.ipynb
│   ├── character-sign-language-detection-EfficientNetB0.ipynb
│   ├── character-sign-language-detection-EfficientNetV2.ipynb
│   ├── character-sign-language-detection-EfficientNetV2L.ipynb
│   ├── character-sign-language-detection-MobileNetV2.ipynb
│   ├── character-sign-language-detection-MobileNetV3Large.ipynb
│   ├── character-sign-language-detection-ResNet50.ipynb
│   ├── character-sign-language-detection-ResNet101.ipynb
│   ├── character-sign-language-detection-ResNet152.ipynb
│   └── ...
├── assets/
│   ├── confusion_matrix/
│   │   ├── ConvNeXt.png
│   │   ├── DenseNet121.png
│   │   ├── ...
│   ├── gradcam_heatmap/
│   ├── lime_visualization/
│   ├── tSNE/
│   ├── validation_accuracy/
│   └── validation_loss/
├── README.md
```

### **Directories**
- **`Notebooks/`**: Contains Jupyter notebooks used for training, evaluating, and visualizing the performance of each model. Each notebook corresponds to a specific CNN architecture.
- **`assets/`**: Stores visualizations and analysis outputs, including:
  - **`confusion_matrix/`**: Class-wise misclassification heatmaps for each model.
  - **`gradcam_heatmap/`**: Grad-CAM visualizations highlighting important regions in input images.
  - **`lime_visualization/`**: LIME visualizations for interpretable predictions.
  - **`tSNE/`**: t-SNE plots showing feature clustering by each model.
  - **`validation_accuracy/`**: Training vs. validation accuracy graphs.
  - **`validation_loss/`**: Training vs. validation loss graphs.

---

## **Dataset**

The dataset used for this study is the **Sign Language Gesture Images Dataset** [13], which consists of **37 unique classes** (alphanumeric gestures). Each class contains 1,500 images, resized to **50x50 pixels**. The dataset is preprocessed and augmented using TensorFlow’s `ImageDataGenerator`.

---

## **Pretrained Models**

The following 15 pretrained CNN architectures were evaluated in this study:

- **EfficientNet Variants**: EfficientNetB0, EfficientNetV2, EfficientNetV2L
- **VGG Variants**: VGG16, VGG19
- **ResNet Variants**: ResNet50, ResNet101, ResNet152
- **MobileNet Variants**: MobileNetV2, MobileNetV3Large
- **DenseNet Variants**: DenseNet121, DenseNet169, DenseNet201
- **ConvNeXt Variants**: ConvNeXt, ConvNeXtBase

All models were initialized with ImageNet weights and fine-tuned for static hand gesture recognition.

---

## **Key Features**

1. **Evaluation Metrics**:
   - **Accuracy**: Top-1, Top-2, and Top-3 accuracy.
   - **Precision, Recall, and F1-Score**: For multi-class classification performance.
2. **Visualization Tools**:
   - Confusion Matrices for misclassification analysis.
   - Grad-CAM for visualizing influential regions in input images.
   - LIME for instance-level interpretability.
   - t-SNE for feature clustering visualization.
3. **Comparative Analysis**:
   - Detailed comparisons across model families (EfficientNet, ResNet, VGG, etc.).
   - Trade-offs between accuracy and computational complexity.

---

## **How to Use**

### **1. Clone the Repository**
```bash
git clone https://github.com/sakhadib/CNN_Comparison_on_Static_Hand_Gesture.git
cd CNN_Comparison_on_Static_Hand_Gesture
```

### **2. Run Notebooks**
Open any notebook in the **`Notebooks/`** directory to train and evaluate a specific model. Ensure you have the required libraries installed (e.g., TensorFlow, PyTorch, OpenCV).

### **3. View Visualizations**
Explore the **`assets/`** directory to review pre-generated visualizations, such as Grad-CAM heatmaps, t-SNE plots, and confusion matrices.

---

## **Results**

The study identifies **ConvNeXt** and **EfficientNetV2L** as the top-performing architectures based on their ability to achieve high accuracy and interpretability.  
Key findings include:
- **ConvNeXt** achieves the highest Top-1 accuracy, demonstrating its robustness for primary predictions.
- **EfficientNet models** balance accuracy with computational efficiency, making them ideal for resource-constrained applications.

---

## **Acknowledgments**

Special thanks to the open-source community for providing the datasets, pretrained models, and visualization tools used in this study.
