"""
Feature engineering for the Severity Score Triage Model.

Two things happen here:
1. Ordinal encoding of naturally ordered categorical fields, and a few
   composite features aligned to the severity factors named in the brief
   (duration, number of failures, vulnerability, financial/emotional/
   physical impact, bereavement, long-term consequences).
2. A clear boundary around the two leakage-risk features identified during
   EDA (EstimatedImpactScore, EstimatedRiskScore) so that a model can be
   built and evaluated both with and without them - see README for the
   reasoning.
"""

import pandas as pd

ORDINAL_MAPS = {
    "EmotionalImpactLevel": {"Minimal": 0, "Low": 1, "Moderate": 2, "Significant": 3, "Severe": 4},
    "PhysicalImpactLevel": {"None": 0, "Minor": 1, "Moderate": 2, "Significant": 3, "Severe": 4},
    "VulnerabilityLevel": {"None": 0, "Low": 1, "Moderate": 2, "High": 3},
    "EvidenceStrength": {"Limited": 0, "Moderate": 1, "Strong": 2},
    "AgeBand": {"Under 18": 0, "18-34": 1, "35-49": 2, "50-64": 3, "65-79": 4, "80+": 5},
}

BOOLEAN_IMPACT_FLAGS = [
    "AnxietyReported",
    "DepressionReported",
    "DeteriorationInHealth",
    "SleepDisruptionReported",
    "LegalChallengePotential",
    "SafeguardingConcern",
    "LongTermImpactFlag",
    "EscalatedInternally",
    "MultipleComplaintThemes",
    "RepeatedFailurePattern",
    "BereavedPerson",
]

# Features flagged during EDA as likely near-duplicates of the target
# (correlation 0.94 and 0.80 respectively, with near-total separation by
# SeverityScore band in boxplots). Described in the data dictionary as
# "internal assessments", suggesting they may be produced by the same
# process that assigns severity - i.e. not genuinely available ahead of
# triage. Kept as an optional feature set so we can compare a model with
# and without them.
LEAKAGE_RISK_COLS = ["EstimatedImpactScore", "EstimatedRiskScore"]


def add_ordinal_encodings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        df[f"{col}_Ord"] = df[col].map(mapping)
    return df


def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Count of negative-impact indicators present for the case - a simple
    # proxy for "how many of the injustice factors listed in the brief
    # apply here".
    flags_as_binary = df[BOOLEAN_IMPACT_FLAGS].apply(lambda s: (s == "Yes").astype(int))
    df["NegativeImpactFlagCount"] = flags_as_binary.sum(axis=1)

    # Combined directly-incurred financial impact (excludes
    # ExpectedFinancialRedressGBP, which is itself a recommendation rather
    # than a measured cost, and correlates fairly strongly with severity -
    # worth treating cautiously for the same reason as the two leakage-risk
    # columns above).
    df["TotalIncurredFinancialImpactGBP"] = (
        df["AdditionalCostsGBP"] + df["DirectFinancialLossGBP"] + df["LostIncomeGBP"]
    )

    # Vulnerability combined with physical/emotional impact - a case
    # involving a highly vulnerable person AND severe impact is worse than
    # either factor alone.
    df["VulnerabilityImpactInteraction"] = (
        df["VulnerabilityLevel_Ord"] * (df["PhysicalImpactLevel_Ord"] + df["EmotionalImpactLevel_Ord"])
    )

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_ordinal_encodings(df)
    df = add_composite_features(df)
    return df


def get_feature_columns(df: pd.DataFrame, include_leakage_risk: bool) -> list:
    """Return the list of columns to use as model features.

    Excludes identifiers, the raw date, the target, and the original text
    versions of columns we ordinal-encoded. Set include_leakage_risk=False
    to build the more conservative, production-realistic feature set.
    """
    exclude = {"CaseReference", "CaseCreatedDate", "SeverityScore", "TriageOutcome"}
    exclude.update(ORDINAL_MAPS.keys())  # keep the _Ord versions instead

    if not include_leakage_risk:
        exclude.update(LEAKAGE_RISK_COLS)

    return [c for c in df.columns if c not in exclude]