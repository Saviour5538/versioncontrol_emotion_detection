import os
import logging
import yaml
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# ------------------------
# Setup logging
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - feature_engineering - %(levelname)s - %(message)s"
)
logger = logging.getLogger("feature_engineering")


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


def save_csv_file(df, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Saved features to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save {file_path}: {e}")
        raise


# ------------------------
# Feature engineering functions
# ------------------------
def load_params(param_file='params.yaml'):
    try:
        with open(param_file, 'r') as f:
            params = yaml.safe_load(f)
        max_features = params['feature_engineering']['max_features']
        logger.info(f"Loaded max_features = {max_features} from {param_file}")
        return max_features
    except Exception as e:
        logger.error(f"Failed to load parameters from {param_file}: {e}")
        raise


def prepare_bow_features(train_texts, test_texts, max_features):
    try:
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(train_texts)
        X_test_bow = vectorizer.transform(test_texts)
        logger.info(f"Bag-of-Words features created with max_features={max_features}")
        return X_train_bow, X_test_bow
    except Exception as e:
        logger.error(f"Failed to create Bag-of-Words features: {e}")
        raise


def create_feature_dataframe(X_bow, labels):
    try:
        df = pd.DataFrame(X_bow.toarray())
        df['label'] = labels
        return df
    except Exception as e:
        logger.error(f"Failed to create feature dataframe: {e}")
        raise


# ------------------------
# Main pipeline
# ------------------------
def main():
    try:
        # Load parameters
        max_features = load_params('params.yaml')

        # Load processed data
        train_df = read_csv_file('./data/processed/train_processed.csv')
        test_df = read_csv_file('./data/processed/test_processed.csv')

        # Extract texts and labels
        X_train = train_df['content'].fillna("").values
        y_train = train_df['sentiment'].values
        X_test = test_df['content'].fillna("").values
        y_test = test_df['sentiment'].values

        # Generate Bag-of-Words features
        X_train_bow, X_test_bow = prepare_bow_features(X_train, X_test, max_features)

        # Create feature DataFrames
        train_features_df = create_feature_dataframe(X_train_bow, y_train)
        test_features_df = create_feature_dataframe(X_test_bow, y_test)

        # Save features
        save_csv_file(train_features_df, './data/features/train_bow.csv')
        save_csv_file(test_features_df, './data/features/test_bow.csv')

        logger.info("Feature engineering completed successfully!")

    except Exception as e:
        logger.exception(f"Feature engineering pipeline failed: {e}")


if __name__ == "__main__":
    main()
