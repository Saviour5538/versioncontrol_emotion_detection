import os
import re
import logging
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("data_preprocessing")


# ------------------------
# Text preprocessing functions
# ------------------------
def lower_case(text):
    return " ".join([w.lower() for w in str(text).split()])


def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([w for w in str(text).split() if w not in stop_words])


def removing_numbers(text):
    return ''.join([i for i in str(text) if not i.isdigit()])


def removing_punctuations(text):
    text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,.-./:;<=>?@[\]^_'{|}~"""), '', str(text))
    text = re.sub(r'\s+', ' ', text)  # remove extra whitespace
    return text.strip()


def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', str(text))


def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w) for w in str(text).split()])


def preprocess_text(text):
    """
    Apply all preprocessing steps sequentially
    """
    try:
        text = lower_case(text)
        text = remove_stop_words(text)
        text = removing_numbers(text)
        text = removing_punctuations(text)
        text = removing_urls(text)
        text = lemmatization(text)
        return text
    except Exception as e:
        logger.error(f"Error processing text: {text} | Error: {e}")
        return ""


# ------------------------
# Data preprocessing pipeline
# ------------------------
def normalize_dataframe(df, text_column='content'):
    """
    Normalize text column in the dataframe
    """
    if text_column not in df.columns:
        logger.error(f"Column '{text_column}' not found in dataframe")
        raise KeyError(f"{text_column} not found")
    
    logger.info(f"Normalizing column: {text_column}")
    df[text_column] = df[text_column].apply(preprocess_text)
    return df


def remove_small_sentences(df, text_column='content', min_words=3):
    """
    Remove rows where text has fewer than `min_words`
    """
    mask = df[text_column].apply(lambda x: len(str(x).split()) >= min_words)
    removed_count = (~mask).sum()
    logger.info(f"Removing {removed_count} small sentences with < {min_words} words")
    return df[mask]


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
        logger.info(f"Saved processed data to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save {file_path}: {e}")
        raise


# ------------------------
# Main preprocessing function
# ------------------------
def main():
    try:
        # File paths
        train_path = './data/raw/train.csv'
        test_path = './data/raw/test.csv'
        processed_train_path = './data/processed/train_processed.csv'
        processed_test_path = './data/processed/test_processed.csv'

        # Load data
        train_df = read_csv_file(train_path)
        test_df = read_csv_file(test_path)

        # Normalize text
        train_df = normalize_dataframe(train_df, text_column='content')
        test_df = normalize_dataframe(test_df, text_column='content')

        # Remove small sentences
        train_df = remove_small_sentences(train_df)
        test_df = remove_small_sentences(test_df)

        # Save processed data
        save_csv_file(train_df, processed_train_path)
        save_csv_file(test_df, processed_test_path)

        logger.info("Data preprocessing completed successfully!")

    except Exception as e:
        logger.exception(f"Data preprocessing pipeline failed: {e}")


if __name__ == "__main__":
    main()
