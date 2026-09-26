# Payment Transaction Anomaly Detection Using Machine Learning

## Project Overview

Payment systems process a large number of transactions every day. Most transactions follow normal patterns, while some transactions may behave differently from the usual pattern.

This project develops a machine learning based system to classify payment transactions as either **Normal** or **Anomalous**.

The project was developed as an educational machine learning prototype and covers the complete workflow from data preparation and exploratory analysis to model development, evaluation and deployment using Streamlit.

---

## Objectives

The main objectives of this project are:

- Understand and explore payment transaction data.
- Identify missing values, duplicate records and class imbalance.
- Perform exploratory data analysis.
- Create useful behavioural and time-based features.
- Build machine learning classification models.
- Evaluate the models using appropriate classification metrics.
- Analyse false positives and false negatives.
- Develop a simple Streamlit application for transaction prediction.

---

## Dataset

The project uses a **synthetic educational transaction dataset** containing 10,000 transaction records.

The target variable is:

- `0` — Normal Transaction
- `1` — Anomalous Transaction

### Target Distribution

- Normal transactions: 9,490
- Anomalous transactions: 510

The dataset is synthetic and was created for educational purposes. It does not represent real bank or customer transaction data.

---

## Features

The dataset contains transaction and behavioural information such as:

- Transaction amount
- Merchant category
- Country
- Home country
- Transaction channel
- Time since previous transaction
- Transaction frequency in the last 24 hours
- Average transaction amount over 30 days
- Account age
- Distance from home
- New device indicator

Additional features were created during preprocessing:

- `amount_ratio`
- `country_changed`
- `high_frequency`
- `transaction_hour`
- `day_of_week`
- `transaction_month`

---

## Machine Learning Approach

The following classification models were explored:

### 1. Logistic Regression

Used as a baseline classification model.

### 2. Decision Tree

Used to capture non-linear relationships between transaction characteristics.

### 3. Random Forest

Used as an ensemble-based classification approach to improve the modelling of complex patterns.

The data was divided into:

- 80% training data
- 20% testing data

Stratified splitting was used because the target classes are imbalanced.

---

## Data Preprocessing

The preprocessing workflow includes:

1. Numerical feature scaling using StandardScaler.
2. Categorical feature encoding using OneHotEncoder.
3. Handling previously unseen categorical values using `handle_unknown="ignore"`.
4. Combining preprocessing and the classifier into a machine learning pipeline.

Keeping preprocessing inside the pipeline ensures that the same transformations are applied during prediction.

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Since anomalous transactions represent a minority class, accuracy alone is not sufficient to understand model performance.

Precision, recall, F1-score and the confusion matrix are therefore particularly important for understanding false alarms and missed anomalies.

---

## Streamlit Application

A Streamlit application was developed to make the trained model easier to use.

The application allows a user to enter transaction characteristics and receive a prediction.

The output includes:

- Normal Transaction or Anomalous Transaction
- Anomaly probability

The application was tested locally in a web browser.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

---

## Project Structure

```text
payment-transaction-anomaly-detection/
│
├── Payment_Transaction_Anomaly_Classification.ipynb
├── payment_transaction_anomaly_dataset.csv
├── payment_anomaly_model.pkl
├── app.py
├── requirements.txt
├── README.md
└── Payment_Transaction_Anomaly_Detection_Project_Report.pdf
