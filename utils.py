"""
=========================================================
Customer Churn Prediction
File : utils.py
Author : Pratiksha Tomar
=========================================================
"""

import os
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


class Utils:

    @staticmethod
    def create_folders():
        """
        Create required folders
        """

        os.makedirs("models", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)

    @staticmethod
    def save_model(model, file_path="models/churn_model.pkl"):
        """
        Save trained model
        """

        Utils.create_folders()

        joblib.dump(model, file_path)

        print(f"\nModel saved successfully at : {file_path}")

    @staticmethod
    def load_model(file_path="models/churn_model.pkl"):
        """
        Load trained model
        """

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Model file not found : {file_path}"
            )

        print("\nModel loaded successfully.")

        return joblib.load(file_path)

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred):
        """
        Save Confusion Matrix
        """

        Utils.create_folders()

        cm = confusion_matrix(y_true, y_pred)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm
        )

        disp.plot()

        plt.title("Confusion Matrix")

        plt.savefig(
            "outputs/confusion_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("Confusion Matrix Saved.")

    @staticmethod
    def plot_roc_curve(model, X_test, y_test):
        """
        Save ROC Curve
        """

        Utils.create_folders()

        RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test
        )

        plt.title("ROC Curve")

        plt.savefig(
            "outputs/roc_curve.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("ROC Curve Saved.")

    @staticmethod
    def print_header(title):
        """
        Print formatted heading
        """

        print("\n" + "=" * 60)
        print(title.center(60))
        print("=" * 60)