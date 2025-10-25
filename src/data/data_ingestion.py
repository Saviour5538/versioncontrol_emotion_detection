
import os

# Compute the absolute path to params.yaml relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(current_dir, "../../params.yaml")  # adjust based on folder structure

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

# ---------------------- Logging Configuration ----------------------

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

# Console Handler (INFO level)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(console_formatter)

# File Handler (DEBUG level)
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, "data_ingestion.log"))
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

# Add handlers (avoid duplicates)
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ---------------------- Helper Functions ----------------------

def load_params(params_path: str) -> float:
    """Load parameters from params.yaml file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
            test_size = params["data_ingestion"]["test_size"]

            if not isinstance(test_size, (float, int)):
                raise ValueError("test_size must be a float or int.")

            logger.info(f"Loaded test_size = {test_size}")
            return test_size

    except FileNotFoundError as e:
        logger.error(f"Parameter file not found: {params_path}")
        raise e
    except KeyError as e:
        logger.error(f"Missing key in params.yaml: {e}")
        raise e
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise e
    except Exception as e:
        logger.exception("Unexpected error while loading params.")
        raise e


def read_data(url: str) -> pd.DataFrame:
    """Read dataset from a URL."""
    try:
        df = pd.read_csv(url)
        if df.empty:
            raise ValueError("Loaded dataframe is empty.")

        logger.info(f"Data successfully read from {url} with shape {df.shape}")
        return df

    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty or unreadable.")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
        raise
    except Exception as e:
        logger.exception(f"Failed to read data from {url}.")
        raise


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and filter data for happiness/sadness sentiments."""
    try:
        logger.info("Starting data processing...")

        if "tweet_id" in df.columns:
            df.drop(columns=["tweet_id"], inplace=True)
            logger.debug("Dropped 'tweet_id' column.")
        else:
            logger.warning("Column 'tweet_id' not found. Skipping drop...")

        if "sentiment" not in df.columns:
            raise KeyError("Column 'sentiment' missing from dataset.")

        final_df = df[df["sentiment"].isin(["happiness", "sadness"])].copy()
        if final_df.empty:
            raise ValueError("No rows found with sentiments 'happiness' or 'sadness'.")

        final_df["sentiment"].replace({"happiness": 1, "sadness": 0}, inplace=True)
        logger.info(f"Processed data shape: {final_df.shape}")

        return final_df

    except KeyError as e:
        logger.error(f"Missing expected column: {e}")
        raise
    except Exception as e:
        logger.exception("Error during data processing.")
        raise


def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Save the train/test datasets to local CSV files."""
    try:
        os.makedirs(data_path, exist_ok=True)

        train_path = os.path.join(data_path, "train.csv")
        test_path = os.path.join(data_path, "test.csv")

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        logger.info(f"Train data saved to: {train_path}")
        logger.info(f"Test data saved to: {test_path}")

    except PermissionError:
        logger.error("Permission denied while saving files.")
        raise
    except Exception as e:
        logger.exception("Error saving train/test data.")
        raise


# ---------------------- Main Function ----------------------

def main() -> None:
    """Main execution flow."""
    try:
        logger.info("Starting data ingestion pipeline...")

        test_size =load_params(params_path)

        df = read_data("https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv")

        final_df = process_data(df)

        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        logger.info(f"Data split into train ({len(train_data)}) and test ({len(test_data)}) samples.")

        data_path = os.path.join("data", "raw")
        save_data(data_path, train_data, test_data)

        logger.info(" Data ingestion completed successfully.")

    except Exception as e:
        logger.exception(" Pipeline failed.")
        raise


# ---------------------- Entry Point ----------------------

if __name__ == "__main__":
    main()
