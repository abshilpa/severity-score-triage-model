# Severity Score Triage Model

A machine learning classification model that predicts complaint severity (`SeverityScore` 1–6) to support
automated case triage.

The model also supports the business-level triage decision:

- **1–3:** Not progressed
- **4–6:** Progressed for further investigation

I prioritised recall of severe cases throughout, because the assessment brief specifies a
low tolerance for severe cases being incorrectly classified as low severity.

---

## Project Objective

My objective was to build a reproducible machine learning pipeline that:

1. Predicts complaint severity from 1 to 6.
2. Supports the binary triage decision of `1–3` versus `4–6`.
3. Minimises the risk of severe cases being missed.
4. Produces interpretable predictions that non-technical stakeholders can understand.
5. Produces predictions for the supplied holdback dataset.

---

## Repository Structure
```text
severity-score-triage-model/
│
├── data/
│   └── raw/
│       ├── training_dataset.csv
│       ├── holdback_dataset.csv
│       └── data_dictionary.csv
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── model_prep.py
│
├── outputs/
│   └── holdback_predictions.csv
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Data

I used two datasets supplied as part of the assessment:

- **Training Dataset** — 2,000 labelled complaint cases containing `SeverityScore`.
- **Holdback Dataset** — 500 unlabelled cases used only for my final predictions.
- **Data Dictionary** — definitions, allowed values, and expected ranges for the
  supplied fields.

I kept the original assessment data unmodified under `data/raw/` and did not edit it
during preprocessing. I also kept the holdback dataset separate from model development —
I never used it for feature selection, hyperparameter tuning, threshold selection, or
model comparison; it was only used once, at the end, to generate final predictions.

## Reproducibility

### Requirements

Python 3.13 is what I used.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

### Running the analysis

Open `notebooks/01_eda.ipynb` in Jupyter or VS Code, select the `.venv` kernel, and run
the notebook from top to bottom. I used a fixed `random_state=42` throughout the modelling
workflow so my results are repeatable.

## My Methodology

### 1. Data Audit and Quality Assessment

I first examined the training and holdback datasets for: dataset dimensions and schema,
data types, missing values, duplicate records, categorical consistency, numerical
distributions, target-class distribution, values outside the ranges specified in the Data
Dictionary, and potential outliers or data-quality anomalies.

I based my data-quality decisions on the supplied Data Dictionary wherever possible,
rather than relying only on statistical outlier detection.

### 2. Missing-Value Treatment

I handled missing values according to their meaning in the supplied Data Dictionary.

The following fields define "None" as a valid category:
- `PrimaryVulnerability`
- `VulnerabilityLevel`
- `PhysicalImpactLevel`
- `FailureTypeSecondary`

When I checked the raw data, the literal value "None" was never actually present in these
fields. I concluded missing values here represent "None / not applicable", and replaced
them with the explicit category `"None"`.

For fields without a valid "None" category, I treated missing values as genuinely missing.
I imputed numerical variables using training-set medians, and categorical variables using
training-set modes, and added missingness indicator flags where appropriate. I learned
every imputation value from the training data only, then applied it unchanged to the
holdback data, to avoid data leakage.

**Holdback-specific missingness:** I found `OmbudsmanInvestigationRequired` is populated
in the training data but 100% missing in the holdback dataset. Since this information
would never be available when predicting the supplied holdback cases, I excluded the
feature from my modelling pipeline entirely, rather than imputing a value that would add
no real signal.

### 3. Schema Documentation Gap

I found two fields present in the supplied datasets that aren't documented in the Data
Dictionary: `AdditionalTreatmentRequired` and `ImpactOnRelationships`. Both are simple
Yes/No fields consistent with the rest of the dataset, so I retained them as categorical
features. I'm recording this documentation gap here rather than silently ignoring it.

### 4. Feature Engineering

I engineered features to reflect the severity factors described in the assessment brief.

**Ordinal features:** I ordinally encoded naturally ordered categorical variables —
`EmotionalImpactLevel`, `PhysicalImpactLevel`, `VulnerabilityLevel`, `EvidenceStrength`,
`AgeBand` — to preserve their ordering rather than treating them as unordered categories.

**Derived features:** I built `NegativeImpactFlagCount`, `TotalIncurredFinancialImpactGBP`,
and `VulnerabilityImpactInteraction` to capture combinations of financial impact,
vulnerability, and physical/emotional impact.

I one-hot encoded the remaining nominal categorical variables, fitting my preprocessing
pipeline on training data and applying the same transformations to validation and holdback
data.

### 5. Leakage Investigation

A key part of my modelling process was identifying features that might contain
information generated after, or during, the severity assessment process itself.

During my exploratory analysis, I found `EstimatedImpactScore` and `EstimatedRiskScore`
showed a very strong relationship with `SeverityScore`. Later, when I ran SHAP on an
initial model, I found `PredictedRemedyBand` and `ExpectedFinancialRedressGBP` were among
the strongest predictors — something my earlier numeric-only correlation check had missed,
since `PredictedRemedyBand` is categorical. All four are described in the Data Dictionary
as internal assessments, estimates, or recommendations.

Because my intended use case is automated triage of new cases, I judged that using
information which might only become available after assessment could produce overly
optimistic results and reduce real-world deployability. I therefore evaluated two feature
sets:

- A **leakage-risk feature set**, including all four columns above.
- A **conservative feature set**, excluding features I considered potentially unavailable
  at the point of triage.

**I selected the conservative feature set as my primary modelling approach.** I'm
retaining the performance difference between the two as an important finding, rather than
using the higher-performing leakage-risk model simply because it produced better
validation metrics.

### 6. Model Development

I compared baseline and tree-based classification approaches, primarily Random Forest and
XGBoost, using stratified validation to preserve the distribution of the six severity
classes.

Random Forest gave me the strongest balance of validation accuracy and severe-case recall
on my conservative feature set, so I selected it for further tuning.

I optimised hyperparameters using `RandomizedSearchCV` (25 parameter combinations, 5-fold
stratified cross-validation), with the search objective prioritising recall of the binary
triage decision rather than accuracy alone. My selected configuration:


### 7. Business-Risk-Aware Evaluation

I decided accuracy alone wasn't sufficient for this problem. Since the brief specifies a
low tolerance for severe cases being classified as low severity, I evaluated both the
six-class prediction task and the operational binary triage decision.

**Six-class evaluation** — I looked at accuracy, macro F1, weighted F1, per-class
precision/recall, and the confusion matrix.

**Binary triage evaluation** — I grouped the six severity classes into 1–3 (Not
progressed) and 4–6 (Progressed), and treated **recall on the Progressed group** as my key
business metric: the proportion of genuinely severe cases successfully flagged for
investigation.

### 8. Threshold Calibration

I compared the default multiclass prediction against a calibrated binary triage threshold:


Approach	Severe-case recall	Severe cases missed	False positives
Default prediction	85.8%	17 / 120	34 / 280
Calibrated threshold	97.5%	3 / 120	87 / 280

I selected a threshold of 0.360 (probability of belonging to the Progressed 4–6 group) as
my operating point. This increased severe-case recall from 85.8% to 97.5%, reducing missed
severe cases from 17 to 3 in my validation set, at the cost of more cases being routed for
further investigation. I consider this an appropriate trade-off given the brief's stated
risk tolerance, where additional review effort is less costly than missing a genuinely
severe case. One consequence worth flagging: on the holdback set, my model predicts 50% of
cases as "Progressed", versus the ~30% base rate I observed in training.

### 9. Final Prediction Strategy

My final prediction process uses a two-stage decision:

- **Stage 1 — Binary triage:** the calibrated probability threshold determines whether a
  case is 1–3 (Not progressed) or 4–6 (Progressed).
- **Stage 2 — Severity score:** I then take the most likely severity class *within* the
  selected band as the final `PredictedSeverityScore`.

This ensures the numeric severity prediction is always consistent with the triage decision
it implies.

### 10. Explainability

I used SHAP (`TreeExplainer`) to investigate the behaviour of my final Random Forest
model. The important drivers I found relate to recovery time, vulnerability, emotional
impact, physical impact, duration of impact, negative-impact indicators, and financial
impact — these align closely with the severity factors described in the assessment brief.
I used this explainability analysis as a model validation tool as much as an output: it let
me assess whether the model's drivers are business-plausible rather than relying solely on
predictive performance, and it's what led me to discover the `PredictedRemedyBand`
leakage risk in the first place.

### 11. Holdback Predictions

After model selection and evaluation, I applied my final preprocessing pipeline and model
to the holdback dataset. As noted above, I never used the holdback dataset for feature
selection, hyperparameter tuning, threshold selection, or model comparison.

My final prediction file contains `CaseReference`, `PredictedSeverityScore`, and
`PredictedTriageOutcome` (the derived binary decision, included for clarity alongside the
required numeric score), generated at `outputs/holdback_predictions.csv`.

### 12. Limitations and What I'd Flag for Production

- **Review workload:** my calibrated threshold prioritises severe-case recall at the cost
  of more cases being referred for investigation. I'd want the operating threshold
  reviewed against the organisation's actual investigation capacity before production
  deployment.
- **Potential leakage assumptions:** I couldn't confirm with the data owner whether
  `EstimatedImpactScore`, `EstimatedRiskScore`, `PredictedRemedyBand`, and
  `ExpectedFinancialRedressGBP` are genuinely available at the point of initial triage —
  my conservative model excludes these features on the assumption they aren't.
- **Class 3 behaviour:** I noticed my calibrated binary decision, by prioritising
  identification of 4–6 cases, rarely predicts class 3 — borderline cases get routed
  towards further investigation rather than treated as low severity. This is expected
  behaviour from my calibration choice, not a bug.
- **Temporal validation:** I didn't perform temporal validation (training on earlier
  cases, testing on later ones), since the supplied data and assessment scope didn't
  establish a clear temporal holdout strategy. I'd revisit this if case-handling practices
  are known to have changed over time.
- **Human oversight:** for production, I'd suggest routing cases close to the decision
  threshold to human review rather than relying entirely on automated classification.

### 13. Future Improvements

If I were to continue this work, I'd look at: temporal validation using future cases,
probability calibration monitoring, explicit cost-sensitive optimisation based on real
operational costs, ongoing monitoring of false-negative rates for severity 4–6, monitoring
for data and concept drift, periodic model retraining, human-in-the-loop review for
borderline cases, and additional validation with domain experts to confirm my leakage
assumptions.
