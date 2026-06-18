# Parkinson's Voice Biomarker Analysis

Machine learning analysis of vocal biomarkers associated with Parkinson's disease using biomedical voice measurements.

## Project Overview

This project explores whether biomedical voice measurements contain measurable patterns associated with Parkinson's disease. The dataset contains acoustic voice features extracted from sustained phonation recordings, including measurements related to pitch, frequency instability, amplitude instability, vocal noise, and nonlinear vocal complexity.

The goal of this project is not to build a clinical diagnostic tool. Instead, the goal is to perform a professional data science workflow that investigates whether vocal biomarkers can distinguish Parkinson's disease recordings from healthy control recordings while accounting for important dataset limitations such as repeated recordings per patient and class imbalance.

## Project Question

**Do voice recordings contain measurable biomarkers associated with Parkinson's disease?**

## Dataset

The project uses the UCI Parkinson's voice dataset.

Dataset source: UCI Machine Learning Repository - Parkinsons Dataset

The dataset includes:

- 195 biomedical voice recordings
- 32 extracted patient IDs from the `name` column
- 22 numeric vocal biomarker features
- Binary target variable: `status`
  - `0` = Healthy
  - `1` = Parkinson's disease

The original documentation describes 31 individuals, but extracting subject IDs from the `name` column produced 32 unique subject labels. This discrepancy is noted as a dataset documentation or subject-label limitation.

## Important Methodological Note

The dataset contains multiple recordings from the same individuals. Because of this, regular random train/test splitting would create data leakage if recordings from the same patient appeared in both the training and testing sets.

To address this, the modeling workflow uses patient-aware validation:

- `GroupShuffleSplit` for the train/test split
- `GroupKFold` for cross-validation
- `patient_id` as the grouping variable

This makes the evaluation more realistic because models are tested on unseen patients rather than repeated recordings from patients they have already seen.

## Notebooks

### 01 Data Cleaning

The cleaning notebook loads the raw Parkinson's voice dataset, checks for missing values and duplicates, extracts a `patient_id` column from the `name` column, validates the target variable, and saves a cleaned dataset for later analysis.

Key cleaning results:

- 195 rows
- No missing values
- No duplicate rows
- 32 extracted patient IDs
- 147 Parkinson's recordings
- 48 healthy recordings

### 02 Exploratory Data Analysis

The EDA notebook explores the dataset at both the recording level and patient level. Since patients have repeated recordings, a patient-level summary dataset is created using the median biomarker value for each patient.

Key EDA findings:

- The dataset is moderately imbalanced toward Parkinson's recordings.
- Frequency features such as `MDVP:Fo(Hz)` and `MDVP:Flo(Hz)` tend to be lower in the Parkinson's group.
- Jitter and shimmer features tend to be higher in the Parkinson's group, suggesting greater vocal instability.
- `NHR` tends to be higher and `HNR` tends to be lower among Parkinson's recordings, suggesting noisier vocal signals.
- Nonlinear features such as `spread1`, `spread2`, and `PPE` show some of the strongest differences between groups.
- Many jitter and shimmer features are highly correlated, indicating feature redundancy.
- PCA shows partial group separation, suggesting useful structure in the biomarker data but not perfect separation.

### 03 Modeling

The modeling notebook evaluates whether vocal biomarkers can classify Parkinson's disease status while avoiding patient-level data leakage.

Models tested:

- Dummy baseline
- Logistic Regression
- Support Vector Machine
- Random Forest
- Gradient Boosting

Evaluation methods:

- Patient-aware train/test split
- Patient-aware grouped cross-validation
- Confusion matrices
- ROC curve
- Precision-recall curve
- Threshold tuning
- Logistic Regression coefficients
- Random Forest feature importance
- Feature group comparison

Evaluation metrics:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

## Modeling Results Summary

Gradient Boosting and Random Forest performed strongest overall, especially compared with the dummy baseline. However, the models still struggled with healthy-class detection, which is important because the dataset is imbalanced and contains a small number of healthy patients.

The modeling results suggest that vocal biomarkers contain useful signal related to Parkinson's status, but performance should be interpreted cautiously due to the small dataset size, repeated patient recordings, and lack of external validation.

## Important Biomarkers

Across EDA and model interpretation, the strongest biomarkers included:

- `PPE`
- `spread1`
- `spread2`
- `RPDE`
- `D2`
- `MDVP:Fo(Hz)`
- `MDVP:Flo(Hz)`
- `MDVP:APQ`
- `MDVP:Jitter(Abs)`
- `MDVP:Jitter(%)`

Nonlinear vocal biomarkers were especially important, which suggests that complex vocal irregularity may provide useful information for distinguishing Parkinson's recordings from healthy recordings.

## Dashboard

This project includes a Streamlit dashboard for interactive exploration of the dataset, biomarker patterns, model results, and project limitations.

Dashboard sections:

- Overview
- Dataset Explorer
- Biomarker Explorer
- Model Performance
- Feature Importance
- Limitations

To run the dashboard:

```bash
streamlit run app.py
```

The dashboard is designed for educational and portfolio purposes only. It is not a medical diagnostic tool.

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/phillipsashton/parkinsons-voice-analysis.git
cd parkinsons-voice-analysis
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebooks in order:

```text
01_data_cleaning.ipynb
02_eda.ipynb
03_modeling.ipynb
```

Run the dashboard:

```bash
streamlit run app.py
```

## Requirements

Core libraries used:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- plotly
- streamlit
- joblib

## Limitations

This project has several important limitations:

- The dataset is small, with only 195 recordings.
- There are only 32 extracted patient IDs.
- The dataset documentation reports 31 individuals, while extracted subject labels show 32 IDs.
- Multiple recordings come from the same patients.
- The dataset is imbalanced toward Parkinson's recordings.
- Healthy-class detection remains limited.
- Many biomarkers are highly correlated, which affects feature importance interpretation.
- The model was not externally validated on an independent dataset.
- Results should not be interpreted as clinical diagnosis.

## Future Work

Potential future improvements include:

- Use nested cross-validation for more rigorous hyperparameter tuning.
- Compare patient-level aggregation strategies such as mean, median, and maximum.
- Evaluate PCA-based dimensionality reduction before classification.
- Test model generalization on an external Parkinson's voice dataset.
- Extend the project to the telemonitoring UPDRS dataset for Parkinson's severity prediction.
- Add dashboard screenshots to the README.
- Deploy the Streamlit dashboard online.

## Citation

If using this dataset, cite the original dataset reference:

Max A. Little, Patrick E. McSharry, Eric J. Hunter, Lorraine O. Ramig.  
"Suitability of dysphonia measurements for telemonitoring of Parkinson's disease."  
IEEE Transactions on Biomedical Engineering, 2008.

## Disclaimer

This project is for educational purposes only. It is not intended for clinical use, medical diagnosis, or treatment decisions.