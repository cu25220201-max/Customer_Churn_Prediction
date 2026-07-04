"""
=========================================================
Customer Churn Prediction
File : evaluate.py
Author : Pratiksha Tomar
=========================================================
"""

import os
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from preprocess import DataPreprocessor


class ModelEvaluator:

    def __init__(self):

        self.processor = DataPreprocessor()

        self.model = joblib.load("models/churn_model.pkl")

        os.makedirs("outputs", exist_ok=True)

    def evaluate(self):

        X_train, X_test, y_train, y_test = self.processor.process()

        
        y_pred = self.model.predict(X_test)

        print("\n" + "=" * 60)
        print("Model Evaluation")
        print("=" * 60)

        print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision : {precision_score(y_test, y_pred):.4f}")
        print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
        print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")

       
        cm = confusion_matrix(y_test, y_pred)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm
        )

        disp.plot()

        plt.title("Confusion Matrix")

        plt.savefig("outputs/confusion_matrix.png")

        plt.close()

       
        RocCurveDisplay.from_estimator(
            self.model,
            X_test,
            y_test
        )

        plt.title("ROC Curve")

        plt.savefig("outputs/roc_curve.png")

        plt.close()

        print("\nGraphs Saved Successfully!")
        print("outputs/confusion_matrix.png")
        print("outputs/roc_curve.png")


if __name__ == "__main__":

    evaluator = ModelEvaluator()

    evaluator.evaluate()