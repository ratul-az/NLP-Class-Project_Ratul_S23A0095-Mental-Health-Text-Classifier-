🧠 Mental Health Text Classification using NLP

**Course:** JIE43303 – Natural Language Processing  
**Student Name:** Ratul  
**Student ID:** S23A0095  
**Project Type:** Academic Class Project  
**Institution:** Universiti Malaysia Kelantan  

---

## 📌 Project Overview

Mental health disorders represent a significant global public health challenge. Early identification of psychological distress through language analysis has the potential to support scalable mental health screening and research. This project investigates the use of **Natural Language Processing (NLP)** techniques for mental health text classification by comparing a traditional machine learning approach with modern Transformer-based models.

The project implements:

- A baseline **TF-IDF + Logistic Regression** classifier  
- A **Transformer-based DistilBERT** model (experimental)  
- A **Flask-based web application** for real-time text prediction and visualization  

The system classifies user-generated text into predefined mental health categories and provides probabilistic confidence scores.

⚠️ **Disclaimer:** This project is for educational and research purposes only and is **not a medical diagnostic tool**.

---

## 🎯 Objectives

- To preprocess and analyze mental health–related textual data  
- To build an interpretable baseline NLP classifier using TF-IDF and Logistic Regression  
- To fine-tune a Transformer-based model (DistilBERT) for improved contextual understanding  
- To evaluate and compare models using accuracy, precision, recall, F1-score, and confusion matrices  
- To deploy the trained baseline model using a Flask web interface  

---

## 🏗️ Project Structure
NLP-Class-Project_-Ratul_-S23A0095-(Mental Health Text Classifier)
│
├── notebooks/
│ └── Mental_Health_Text_Classification.ipynb
│
├── dataset/
│ └── mental_health_dataset.csv
│
├── templates/
│ └── index.html
│
├── app.py
├── requirements.txt
└── README.md



---

## 📓 Notebook Description

The notebook located in the `notebooks/` directory documents the **entire experimental pipeline**, including:

- Dataset exploration and class distribution analysis  
- Text preprocessing strategy  
- TF-IDF feature extraction  
- Logistic Regression model training and evaluation  
- Confusion matrix and metric analysis  
- DistilBERT fine-tuning (experimental)  
- Model serialization using `joblib`  

This notebook serves as the **research and experimentation component** of the project.

---

## 🌐 Web Application (Flask)

A Flask-based web application is implemented to demonstrate real-time inference using the trained baseline model.

### Features

- User-friendly modern UI  
- Text input for mental health prediction  
- Predicted label display  
- Animated confidence meter  
- Class-wise probability breakdown  
- Color-coded output based on predicted class  

### Run the Application Locally

bash
pip install -r requirements.txt
python app.py

Then open:
http://127.0.0.1:5000

## 📊 Models Used

### 1️⃣ Baseline Model
- TF-IDF Vectorizer  
- Logistic Regression  

**Advantages**
- Interpretable  
- Computationally efficient  
- Suitable as a strong baseline model  

---

### 2️⃣ Transformer Model (Experimental)
- DistilBERT  

**Advantages**
- Contextual language understanding  
- Improved recall for minority mental health classes  

**Limitations**
- Higher computational cost  
- Reduced interpretability  

---

## 📈 Evaluation Metrics

Models are evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

This multi-metric evaluation ensures reliable performance assessment, particularly under class imbalance conditions.

---

## 📂 Dataset Information

- The dataset used in this project was obtained from Kaggle and is publicly available.
- For academic reproducibility and lecturer evaluation, the dataset has been included in the `dataset/` directory of this repository.
- All preprocessing, training, and evaluation steps directly reference this dataset as documented in the notebook.

---

## 🔁 Model Artifacts and Reproducibility

- Due to GitHub file size limitations, trained model artifacts (`.joblib` files) are not included in this repository.

**To reproduce the trained models:**
- Open `notebooks/Mental_Health_Text_Classification.ipynb`
- Run all cells sequentially
- The trained TF-IDF vectorizer and Logistic Regression model will be automatically saved locally in an `artifacts/` directory
- After this step, the Flask application (`app.py`) can be executed normally  

This approach ensures full reproducibility while adhering to repository size constraints.

---

## ⚠️ Ethical Disclaimer

- This project is intended solely for academic and research purposes.
- It is not designed for clinical diagnosis or medical decision-making.

For real-world deployment, additional safeguards would be required, including:
- Clinical validation  
- Bias and fairness evaluation  
- Privacy protection  
- Explainability mechanisms  

---

## 🚀 Future Work

Potential future extensions include:
- Dataset expansion and cross-domain validation  
- Advanced class imbalance mitigation techniques  
- Multimodal mental health analysis  
- Explainable AI integration  
- Secure and scalable deployment  

---

## 🛠️ Technologies Used

- Python  
- scikit-learn  
- TensorFlow  
- Hugging Face Transformers  
- Flask  
- HTML / CSS  
- Google Colab  

---

## 📜 License

- This project is released for educational use only.
