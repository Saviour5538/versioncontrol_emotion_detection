import os
import logging
import pickle
import yaml
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# ------------------------
# Setup logging
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - model_building - %(levelname)s - %(message)s"
)
logger = logging.getLogger("model_building")

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

def save_model(model, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved model to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save model to {file_path}: {e}")
        raise

# ------------------------
# Model functions
# ------------------------
def load_params(param_file='params.yaml'):
    try:
        with open(param_file, 'r') as f:
            params = yaml.safe_load(f)
        model_params = params['model_building']
        logger.info(f"Loaded model parameters: {model_params}")
        return model_params
    except Exception as e:
        logger.error(f"Failed to load parameters from {param_file}: {e}")
        raise

def train_model(X, y, n_estimators, learning_rate):
    try:
        clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        clf.fit(X, y)
        logger.info("Model training completed successfully")
        return clf
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        raise

# ------------------------
# Main pipeline
# ------------------------
def main():
    try:
        # Load parameters
        params = load_params()

        # Load training data
        train_df = read_csv_file('./data/features/train_tfidf.csv')
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values

        # Train model
        model = train_model(X_train, y_train, params['n_estimators'], params['learning_rate'])

        # Save model
        save_model(model, './model.pkl')

        logger.info("Model building pipeline completed successfully!")

    except Exception as e:
        logger.exception(f"Model building pipeline failed: {e}")

if __name__ == "__main__":
    main()
