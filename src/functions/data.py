"""
data.py
=======

Provides data processing utilities for the Banking Agent Demo application. This module includes MLRun handler functions for:

- Returning a copy of a provided DataFrame.
- Performing sentiment analysis on text data using HuggingFace transformer models and appending sentiment labels and scores to the DataFrame.
- Preprocessing data for machine learning by splitting into train/test sets, encoding categorical features (including sentiment), and optionally dropping columns.

These functions are designed to be used in MLRun pipelines for preparing and analyzing banking-related datasets, supporting downstream tasks such as sentiment analysis and churn prediction.
"""

import mlrun
# import pandas
import pandas as pd
# import pandera as pa
from datasets import Dataset #, load_dataset
# from pandera.typing import DataFrame
from sklearn import set_config
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from transformers import AutoTokenizer, RobertaForSequenceClassification, pipeline

set_config(transform_output="pandas")


@mlrun.handler(outputs=["data"])
def get_data(data: pd.DataFrame):
    """
    Returns a copy of the provided DataFrame.

    :param data: The input pandas DataFrame.

    :returns: A copy of the input DataFrame.
    """
    return data.copy()


@mlrun.handler(outputs=["data_w_sentiment"])
def sentiment_analysis(data: pd.DataFrame, sentiment_model: str, text_column: str):
    """
    Performs sentiment analysis on a pandas DataFrame using a specified HuggingFace transformer model.

    This function applies a sentiment analysis pipeline to the text in the specified column of the input DataFrame,
    returning a new DataFrame with additional columns for sentiment label and sentiment score.

    :param data: Input pandas DataFrame containing the text data.
    :param sentiment_model: HuggingFace model name or path for sentiment analysis (e.g., 'cardiffnlp/twitter-roberta-base-sentiment-latest').
    :param text_column: Name of the column in the DataFrame containing the text to analyze.

    :returns: A pandas DataFrame with the original data and additional columns:
             - 'sentiment_label': The predicted sentiment label for each row.
             - 'sentiment_score': The confidence score for the predicted sentiment.
    """
    tokenizer = AutoTokenizer.from_pretrained(sentiment_model)
    model = RobertaForSequenceClassification.from_pretrained(sentiment_model)
    sentiment_classifier = pipeline(
        task="sentiment-analysis",
        tokenizer=tokenizer,
        model=model,
        top_k=1,
    )

    def sentiment(rows):
        resp = sentiment_classifier(rows[text_column])
        return {
            "sentiment_label": [i[0]["label"] for i in resp],
            "sentiment_score": [i[0]["score"] for i in resp],
        }

    data_w_sentiment = Dataset.from_pandas(data).map(
        sentiment, batched=True, batch_size=50
    )

    return data_w_sentiment.to_pandas()


@mlrun.handler(outputs=["train", "test", "preprocessor:object"])
def process_data(
    data: pd.DataFrame,
    label_column: str,
    test_size: float,
    sentiment_column: str,
    ordinal_columns: list = None,
    drop_columns: list = None,
    random_state: int = 42,
):
    """
    Preprocesses a DataFrame for machine learning by splitting into train/test sets, encoding categorical features,
    and optionally dropping columns.

    This function splits the input data into training and testing sets, encodes specified ordinal and sentiment columns,
    optionally drops specified columns, and returns the transformed train and test sets along with the fitted preprocessor.

    :param data: Input DataFrame containing features and label.
    :param label_column: Name of the column to use as the label.
    :param test_size: Fraction of data to use for the test set.
    :param sentiment_column: Name of the column containing sentiment labels to be ordinally encoded.
    :param ordinal_columns: List of column names to be ordinally encoded (default: None).
    :param drop_columns: List of column names to drop from the data (default: None).
    :param random_state: Random seed for reproducibility (default: 42).

    :returns: Tuple of (train DataFrame, test DataFrame, fitted preprocessor pipeline).
    """
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)

    # Remove label before transforming
    y_train = train.pop(label_column)
    y_test = test.pop(label_column)

    ordinal_columns = ordinal_columns or []
    preprocessor = make_pipeline(
        make_column_transformer(
            # (OneHotEncoder(sparse_output=False), ["state"]),
            (OrdinalEncoder(), ordinal_columns),
            (
                OrdinalEncoder(categories=[["negative", "neutral", "positive"]]),
                [sentiment_column],
            ),
            ("drop", drop_columns),
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
    )

    preprocessor.fit(train)
    train = preprocessor.transform(train)
    test = preprocessor.transform(test)

    # Re-add label after transforming
    train[label_column] = y_train
    test[label_column] = y_test

    return train, test, preprocessor
