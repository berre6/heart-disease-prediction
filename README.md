# Heart Disease Prediction using Machine Learning

## 📌 Project Overview
This project focuses on predicting the presence of heart disease using clinical patient data and machine learning techniques.  
It was developed as part of my preparation for graduate studies in **Biomedical Engineering**, with an emphasis on applying data science to healthcare problems.

## 🫀 Motivation
Cardiovascular diseases are among the leading causes of death worldwide.  
Early detection using clinical indicators can significantly improve patient outcomes.  
This project demonstrates how machine learning can assist in medical decision support systems.

## 📊 Dataset
- Source: UCI Heart Disease Dataset (Cleveland)
- Number of samples after cleaning: 299
- Target variable:
  - `has_disease = 1` → presence of heart disease  
  - `has_disease = 0` → no heart disease

## 🧪 Features Used
- Age
- Resting blood pressure (`trestbps`)
- Cholesterol level (`chol`)
- Maximum heart rate (`thalch`)
- ST depression (`oldpeak`)
- Number of major vessels (`ca`)

## ⚙️ Methodology
- Data cleaning (handling missing values)
- Feature selection
- Train-test split (80% train / 20% test)
- Model: **Logistic Regression**
- Evaluation metric: **Accuracy**

## 📈 Results
- Model Accuracy: **78.3%**
- The model shows a reasonable performance for a baseline medical classification task.

## 🧠 Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Git & GitHub

## 🚀 Future Improvements
- Add ROC curve and confusion matrix
- Try advanced models (Random Forest, XGBoost)
- Feature scaling and hyperparameter tuning
- Clinical interpretation of feature importance

## 🎓 Academic Relevance
This project reflects my interest in combining **machine learning and biomedical data analysis**, and serves as an introductory study for graduate-level research in biomedical engineering and health informatics.

## Model Comparison

In this project, multiple machine learning models were trained and evaluated
on the same heart disease dataset to compare their predictive performance.

### Models Used
- Logistic Regression (baseline model)
- K-Nearest Neighbors (KNN)
- Random Forest Classifier

### Features
The following clinical features were used:
- Age
- Resting blood pressure (trestbps)
- Serum cholesterol (chol)
- Maximum heart rate achieved (thalch)
- ST depression (oldpeak)
- Number of major vessels (ca)

### Evaluation Metric
- Accuracy on the test set (80/20 train-test split)

### Results
The results show that different models achieve different accuracy levels,
highlighting the importance of model selection in medical prediction tasks.
Ensemble-based models such as Random Forest generally outperform simpler
baseline models.

This comparative approach provides insight into the trade-offs between
model complexity and predictive performance in biomedical applications.

