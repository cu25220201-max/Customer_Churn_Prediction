"""
=========================================================
Customer Churn Prediction
File : train.py
Author : Pratiksha Tomar
=========================================================
"""

import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from preprocess import DataPreprocessor


class ModelTrainer:

    def __init__(self):

        self.processor = DataPreprocessor()

        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=42
            )
        }

    def train(self):

        X_train, X_test, y_train, y_test = self.processor.process()

        best_model = None
        best_accuracy = 0
        best_name = ""

        print("\n" + "=" * 60)
        print("Training Models")
        print("=" * 60)

        for name, model in self.models.items():

            print(f"\nTraining {name}...")

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)

            print(f"Accuracy : {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_name = name

        os.makedirs("models", exist_ok=True)

        joblib.dump(best_model, "models/churn_model.pkl")
        joblib.dump(self.processor.scaler, "models/scaler.pkl")

        print("\n" + "=" * 60)
        print("Training Completed Successfully")
        print("=" * 60)
        print(f"Best Model : {best_name}")
        print(f"Best Accuracy : {best_accuracy:.4f}")
        print("Saved : models/churn_model.pkl")
        print("Saved : models/scaler.pkl")

        return best_model


if __name__ == "__main__":

    trainer = ModelTrainer()
    trainer.train()