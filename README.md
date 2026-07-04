# Customer Churn Prediction

## Project Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to leave a company's service based on historical customer data.

This project uses multiple machine learning algorithms to train and evaluate models, then selects the best-performing model for prediction.

---

## Features

- Data Preprocessing
- Data Cleaning
- Feature Encoding
- Feature Scaling
- Train/Test Split
- Multiple Machine Learning Models
- Model Evaluation
- Customer Churn Prediction
- Confusion Matrix
- ROC Curve
- Model Saving & Loading

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

---

## Project Structure

```
Customer_Churn_Prediction/
│
├── dataset/
│   └── churn.csv
│
├── models/
│   └── churn_model.pkl
│
├── notebooks/
│   └── EDA.ipynb
│
├── outputs/
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

This project uses the **Telco Customer Churn Dataset**.

Dataset includes customer information such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Contract
- Payment Method
- Monthly Charges
- Total Charges
- Churn Status

---

## Machine Learning Models

The following algorithms are used:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

The best-performing model is automatically saved.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Customer_Churn_Prediction.git
```

Go to project folder:

```bash
cd Customer_Churn_Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

Run the main application:

```bash
python app.py
```

---

## Menu

```
1. Train Model
2. Evaluate Model
3. Predict Customer Churn
4. Exit
```

---

## Output

### Model Training

- Trains multiple ML models
- Compares accuracy
- Saves best model

### Evaluation

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve

### Prediction

Predicts whether a customer will:

- Churn
- Not Churn

---

## Output Files

```
outputs/

confusion_matrix.png

roc_curve.png
```

---

## Future Improvements

- XGBoost Integration
- LightGBM
- Hyperparameter Tuning
- Streamlit Web App
- Flask API Deployment
- Feature Importance Visualization
- SHAP Explainability
- Cross Validation

---

## Author

**Pratiksha Tomar**

Machine Learning Enthusiast

---

## License

This project is developed for educational and internship purposes.