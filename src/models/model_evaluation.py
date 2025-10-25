import numpy as np
import pandas as pd
import pickle
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# ------------------------
# Setup logging
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - model_evaluation - %(levelname)s - %(message)s"
)
logger = logging.getLogger("model_evaluation")


# ------------------------
# File I/O functions
# ------------------------
def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        raise


def load_model(model_path):
    try:
        model = pickle.load(open(model_path, 'rb'))
        logger.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise


def save_metrics(metrics_dict, file_path):
    try:
        with open(file_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        logger.info(f"Saved evaluation metrics to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics to {file_path}: {e}")
        raise


# ------------------------
# Evaluation functions
# ------------------------
def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics_dict = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }

        logger.info(f"Evaluation metrics: {metrics_dict}")
        return metrics_dict
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise


# ------------------------
# Main pipeline
# ------------------------
def main():
    try:
        # Load test data and model
        test_df = read_csv_file('./data/features/test_bow.csv')
        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values

        model = load_model('model.pkl')

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Save metrics
        save_metrics(metrics, 'metrics.json')

        logger.info("Model evaluation completed successfully!")

    except Exception as e:
        logger.exception(f"Model evaluation pipeline failed: {e}")


if __name__ == "__main__":
    main()
