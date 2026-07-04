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


## Screenshots

<img width="952" height="452" alt="image" src="https://github.com/user-attachments/assets/ce89ef58-6dc7-4ab3-953a-0573744c3e07" />

<img width="957" height="458" alt="image" src="https://github.com/user-attachments/assets/0f8cf4b9-8fed-419f-af38-08bd2afa7041" />


<img width="956" height="431" alt="image" src="https://github.com/user-attachments/assets/631878ec-63ac-444e-8010-8c2c9b7a9b8c" />


<img width="944" height="391" alt="image" src="https://github.com/user-attachments/assets/00938c50-81c1-4e8e-b660-176a210a99ba" />

<img width="940" height="365" alt="image" src="https://github.com/user-attachments/assets/08a5cd92-889e-42b9-971c-1578581ac005" />

<img width="960" height="362" alt="image" src="https://github.com/user-attachments/assets/528cbd8e-42a1-4e42-912e-431909f992fc" />

<img width="960" height="377" alt="image" src="https://github.com/user-attachments/assets/b6830430-4acb-4d7a-8cce-3de2d345791b" />


<img width="955" height="368" alt="image" src="https://github.com/user-attachments/assets/1248d709-2cb4-4d3e-9c55-f16700004868" />





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

## 🎯 Internship Project

This project was successfully developed as part of the **CodeSoft Machine Learning Internship Program** to demonstrate practical implementation of Machine Learning, Data Preprocessing, Model Evaluation, and Interactive Web Application Development using Streamlit.

---


## Author

**Pratiksha Tomar**

Machine Learning Enthusiast

---

## License

This project is developed for educational and internship purposes.
