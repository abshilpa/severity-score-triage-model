"""
Data cleaning utilities for the Severity Score Triage Model.

Key principle: any statistic used for imputation (median, mode) must be
learned from the training set only, then applied identically to both the
training and holdback sets. This avoids leaking holdback information into
the pipeline and mirrors how the model would behave in production, where
only historical data is available at fit time.
"""

import pandas as pd

# Columns where the data dictionary defines "None" as a valid category,
# and missing values in the raw data represent that category rather than
# genuinely absent information.
NONE_CATEGORY_COLS = [
    "PrimaryVulnerability",
    "VulnerabilityLevel",
    "PhysicalImpactLevel",
    "FailureTypeSecondary",
]

# Columns with genuinely missing values (no "None" category applies).
GENUINE_MISSING_NUMERIC = [
    "ExpectedFinancialRedressGBP",
    "RecoveryTimeMonths",
    "AdditionalCostsGBP",
    "EstimatedImpactScore",
]

GENUINE_MISSING_CATEGORICAL = [
    "Jurisdiction",
    "EvidenceStrength",
    "OrganisationType",
    "EmotionalImpactLevel",
    "ServiceArea",
]

# Columns dropped entirely: present in training but 100% missing in the
# holdback set, meaning they are unavailable at prediction time and cannot
# be part of a genuinely production-usable model.
UNAVAILABLE_AT_PREDICTION_TIME = [
    "OmbudsmanInvestigationRequired",
]


def fit_impute_values(train_df: pd.DataFrame) -> dict:
    """Learn imputation values (median / mode) from the training set only."""
    impute_values = {}
    for col in GENUINE_MISSING_NUMERIC:
        impute_values[col] = train_df[col].median()
    for col in GENUINE_MISSING_CATEGORICAL:
        impute_values[col] = train_df[col].mode(dropna=True).iloc[0]
    return impute_values


def clean_data(df: pd.DataFrame, impute_values: dict) -> pd.DataFrame:
    """Apply cleaning steps to a dataframe (train or holdback).

    - Parses CaseCreatedDate to a real datetime.
    - Drops columns unavailable at prediction time (see
      UNAVAILABLE_AT_PREDICTION_TIME).
    - Fills "None"-category columns with the literal string "None".
    - Adds a *_WasMissing flag for genuinely missing columns before imputing,
      since missingness itself may be informative.
    - Imputes genuinely missing numeric/categorical columns using values
      learned from the training set (passed in via impute_values).
    """
    df = df.copy()

    df["CaseCreatedDate"] = pd.to_datetime(df["CaseCreatedDate"], format="%d-%m-%Y")

    df = df.drop(columns=[c for c in UNAVAILABLE_AT_PREDICTION_TIME if c in df.columns])

    for col in NONE_CATEGORY_COLS:
        df[col] = df[col].fillna("None")

    for col in GENUINE_MISSING_NUMERIC + GENUINE_MISSING_CATEGORICAL:
        df[f"{col}_WasMissing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(impute_values[col])

    return df