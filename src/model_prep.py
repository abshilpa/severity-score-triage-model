"""
Final feature matrix preparation: converts remaining text columns (nominal
categoricals and Yes/No booleans) into numeric form via one-hot encoding,
fit on the training set and applied identically to the holdback set so
column sets always match.
"""

import pandas as pd


def build_feature_matrix(train_df: pd.DataFrame, holdback_df: pd.DataFrame, feature_cols: list):
    """One-hot encode remaining text columns; align holdback columns to train.

    Returns (X_train, X_holdback) as numeric dataframes with identical
    columns in identical order.
    """
    train_X = train_df[feature_cols].copy()
    holdback_X = holdback_df[feature_cols].copy()

    text_cols = train_X.select_dtypes(include=["object", "string"]).columns.tolist()

    train_encoded = pd.get_dummies(train_X, columns=text_cols, dummy_na=False)
    holdback_encoded = pd.get_dummies(holdback_X, columns=text_cols, dummy_na=False)

    # Align columns: any category seen in train but not holdback (or vice
    # versa) is filled with 0 rather than dropped, so both matrices have
    # identical columns in identical order.
    train_encoded, holdback_encoded = train_encoded.align(
        holdback_encoded, join="outer", axis=1, fill_value=0
    )

    return train_encoded, holdback_encoded