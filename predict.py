"""
=========================================================
Customer Churn Prediction
File : predict.py
Author : Pratiksha Tomar
=========================================================
"""

import joblib
import pandas as pd


class CustomerPredictor:

    def __init__(self):

        self.model = joblib.load("models/churn_model.pkl")

        df = pd.read_csv("dataset/churn.csv")

        if "customerID" in df.columns:
            df.drop("customerID", axis=1, inplace=True)

    
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

        df["TotalCharges"] = df["TotalCharges"].fillna(
            df["TotalCharges"].median()
        )

        df = pd.get_dummies(df, drop_first=True)

    
        self.feature_columns = df.drop(
            columns=["Churn_Yes"]
        ).columns


    def get_input(self):

        print("\nEnter Customer Details\n")

        data = {}

        data["gender"] = input("Gender (Male/Female): ")
        data["SeniorCitizen"] = int(input("Senior Citizen (0/1): "))
        data["Partner"] = input("Partner (Yes/No): ")
        data["Dependents"] = input("Dependents (Yes/No): ")
        data["tenure"] = int(input("Tenure: "))
        data["PhoneService"] = input("Phone Service (Yes/No): ")
        data["MultipleLines"] = input(
            "Multiple Lines (Yes/No/No phone service): "
        )
        data["InternetService"] = input(
            "Internet Service (DSL/Fiber optic/No): "
        )
        data["OnlineSecurity"] = input(
            "Online Security (Yes/No/No internet service): "
        )
        data["OnlineBackup"] = input(
            "Online Backup (Yes/No/No internet service): "
        )
        data["DeviceProtection"] = input(
            "Device Protection (Yes/No/No internet service): "
        )
        data["TechSupport"] = input(
            "Tech Support (Yes/No/No internet service): "
        )
        data["StreamingTV"] = input(
            "Streaming TV (Yes/No/No internet service): "
        )
        data["StreamingMovies"] = input(
            "Streaming Movies (Yes/No/No internet service): "
        )
        data["Contract"] = input(
            "Contract (Month-to-month/One year/Two year): "
        )
        data["PaperlessBilling"] = input(
            "Paperless Billing (Yes/No): "
        )
        data["PaymentMethod"] = input(
            "Payment Method: "
        )
        data["MonthlyCharges"] = float(
            input("Monthly Charges: ")
        )
        data["TotalCharges"] = float(
            input("Total Charges: ")
        )

        customer = pd.DataFrame([data])

        
        customer = pd.get_dummies(customer)

        
        for col in self.feature_columns:
            if col not in customer.columns:
                customer[col] = 0

        customer = customer[self.feature_columns]

        return customer


    def predict(self):

        customer = self.get_input()

        prediction = self.model.predict(customer)[0]

        probability = self.model.predict_proba(customer)[0]

        print("\n" + "=" * 60)

        if prediction == 1:
            print("Prediction : Customer Will Churn")
        else:
            print("Prediction : Customer Will NOT Churn")

        print(f"Confidence : {max(probability)*100:.2f}%")

        print("=" * 60)


if __name__ == "__main__":

    predictor = CustomerPredictor()

    predictor.predict()