"""
=========================================================
Customer Churn Prediction
File : preprocess.py
Author : Pratiksha Tomar
=========================================================
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:

    def __init__(self, file_path="dataset/churn.csv"):
        self.file_path = file_path
        self.scaler = StandardScaler()


    def load_data(self):
        """
        Load dataset
        """

        df = pd.read_csv(self.file_path)

        print("\nDataset Loaded Successfully!")
        print("Shape :", df.shape)
        print("\nColumns :")
        print(df.columns.tolist())

        return df

    def clean_data(self, df):
        """
        Clean dataset
        """

        if "customerID" in df.columns:
            df.drop("customerID", axis=1, inplace=True)

        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"],
                errors="coerce"
            )

        
            df["TotalCharges"] = df["TotalCharges"].fillna(
                df["TotalCharges"].median()
            )

        return df

    def encode_features(self, df):
        """
        Encode categorical columns
        """

        df = pd.get_dummies(df, drop_first=True)

        return df

    def split_data(self, df):
        """
        Split dataset
        """

        
        if "Churn_Yes" in df.columns:
            target = "Churn_Yes"
        elif "Churn" in df.columns:
            target = "Churn"
        else:
            raise ValueError(
                f"Churn column not found.\nAvailable columns:\n{df.columns.tolist()}"
            )

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        return X_train, X_test, y_train, y_test


    def scale_features(self, X_train, X_test):
        """
        Feature Scaling
        """

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test


    def process(self):
        """
        Complete preprocessing pipeline
        """

        print("=" * 60)
        print("Loading Dataset...")
        print("=" * 60)

        df = self.load_data()

        df = self.clean_data(df)

        df = self.encode_features(df)

        X_train, X_test, y_train, y_test = self.split_data(df)

        X_train, X_test = self.scale_features(
            X_train,
            X_test
        )

        print("\nPreprocessing Completed Successfully!")
        print("Training Samples :", len(X_train))
        print("Testing Samples :", len(X_test))

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )

if __name__ == "__main__":

    processor = DataPreprocessor()

    X_train, X_test, y_train, y_test = processor.process()

    print("\nDone.")