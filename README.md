---

# 🌱 Crop Disease Detection using Deep Learning

## 📜 Abstract

The agricultural sector is crucial for global food security, but plant diseases threaten crop yields and biodiversity. This project leverages deep learning to provide a solution for the timely and accurate detection of crop diseases. By utilizing the **ResNet29** model on the extensive **Plant Village** dataset, we aim to enhance disease diagnosis accuracy, ensuring agricultural sustainability. Our approach involves rigorous experimentation, data preprocessing, model training, and evaluation, showcasing the effectiveness of deep learning in identifying 38 distinct crop disease classes.

## 🧠 Key Features

- **Deep Learning Model:** Implementation of the ResNet29 model, a state-of-the-art deep learning architecture, optimized for disease detection.
- **Extensive Dataset:** Utilization of the Plant Village dataset, comprising 54,303 images across 38 different classes, ensuring robust model training and evaluation.
- **High Accuracy:** Achieved a testing accuracy of up to 95.35% using a carefully designed training regimen.
- **Automated Disease Detection:** Minimizes the need for manual observation, offering a scalable solution for real-world agricultural applications.

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- TensorFlow or PyTorch
- Kaggle API (for dataset access)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sarangs1621/crop-disease-detection.git
   cd crop-disease-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset:**
   - The dataset can be obtained from [Kaggle](https://www.kaggle.com/datasets) by searching for "Plant Village".
   - Alternatively, you can find it on [GitHub](https://github.com/spMohanty/PlantVillage-Dataset).

### Usage

1. **Preprocess the Data:**
   - The images are resized to 64x64 pixels.
   - Labels are encoded and converted using one-hot encoding.

2. **Train the Model:**
   - Train the model with different batch sizes (32, 64, 128) and observe the performance.
   - Use the Adam optimizer and Categorical Cross-Entropy loss function.

3. **Evaluate the Model:**
   - Test the model on the validation and testing datasets.
   - Early stopping is implemented to avoid overfitting.

### Results

- **Training Accuracy:** Up to 98.52%
- **Validation Accuracy:** Up to 95.35%
- **Testing Accuracy:** Up to 95.08%

Visual representations of training and validation losses and accuracies are available for different batch sizes.

## 📊 Results and Discussion

The ResNet29 model demonstrated remarkable precision, achieving a 95.35% accuracy in identifying 38 distinct crop diseases. By adjusting key parameters such as batch size and the number of epochs, the model's robustness was validated across different sample configurations. This project underscores the potential of deep learning in revolutionizing agricultural practices by providing efficient, scalable disease detection systems.

| Batch Size | Training Accuracy | Validation Accuracy | Testing Accuracy |
|------------|-------------------|---------------------|------------------|
| 32         | 98.45%            | 94.45%              | 94.62%           |
| 64         | 97.94%            | 95.08%              | 95.35%           |
| 128        | 98.52%            | 95.20%              | 94.94%           |

## 📚 References

1. **Plant Village Dataset:** Available on [Kaggle](https://www.kaggle.com/datasets) and [GitHub](https://github.com/spMohanty/PlantVillage-Dataset).
2. **Research Paper:** [Plant Disease Detection with Deep Learning](https://doi.org/10.4236/jcc.2020.86002).

## 🤝 Contributions

Feel free to fork this repository, submit issues, and send pull requests. Any improvements and suggestions are welcome!

## 📬 Contact

If you have any questions or want to discuss the project, reach out via email at [sarangsnair1621@gmail.com](mailto:sarangsnair1621@gmail.com).

---
