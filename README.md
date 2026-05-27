# The-Impact-of-Missing-Clinical-Data-on-Multimodal-Alzheimer-s-Disease-Classification

# Improving Robustness of Multimodal Alzheimer’s Disease Classification Under Missing Clinical Data

## Overview

This project investigates how multimodal machine learning systems for Alzheimer’s disease diagnosis behave when clinical information becomes incomplete or unavailable. While many Alzheimer’s classification studies assume perfectly complete patient records, real-world healthcare systems rarely contain fully complete data. Cognitive testing may be partially administered, demographic information may be unavailable, and electronic medical records often contain missing fields.

The central goal of this project was therefore not simply to maximize classification accuracy, but to evaluate how robust multimodal models remain when one source of information disappears.

This study combines:

* Clinical and cognitive features
* Structural MRI biomarkers
* Missing modality simulation
* Robustness training through modality dropout
* Cross-validated evaluation

The final results demonstrate that multimodal models can become highly dependent on clinical information, but that robustness training substantially improves resilience to missing data while maintaining nearly identical baseline predictive performance.

---

# Research Motivation

Alzheimer’s disease is one of the most important neurodegenerative disorders worldwide. Early diagnosis is clinically valuable because:

* treatment interventions are more effective earlier in disease progression
* patients and families can better plan long-term care
* clinicians can monitor disease progression sooner
* enrollment into therapeutic trials improves

Machine learning systems have increasingly been used to assist Alzheimer’s classification by integrating:

* cognitive testing
* demographic information
* MRI imaging
* biomarkers
* longitudinal patient data

However, many existing machine learning studies assume complete patient information. In real clinical settings, this assumption is unrealistic.

Examples of real-world missingness include:

* incomplete cognitive testing
* unavailable MRI scans
* missing demographic variables
* partially transferred medical records
* inconsistent healthcare documentation

A clinically deployable healthcare AI system must therefore remain stable even when parts of the patient record are missing.

This project specifically investigates:

> How does multimodal Alzheimer’s disease classification degrade under missing clinical information, and can modality dropout training improve robustness to missing data?

---

# Dataset

## Alzheimer’s Disease Neuroimaging Initiative (ADNI)

This project uses data from the Alzheimer’s Disease Neuroimaging Initiative (ADNI), one of the largest and most widely used public Alzheimer’s research datasets.

ADNI contains:

* demographic information
* cognitive testing
* MRI imaging
* PET imaging
* longitudinal follow-up
* clinical diagnosis labels

The study focused specifically on:

* Cognitively Normal (CN) patients
* Dementia (DEM) patients

creating a binary classification task.

---

# Modalities Used

## Clinical Modality

The clinical modality captures observable behavioral and cognitive performance.

Final clinical variables:

| Feature  | Biological Meaning                                               |
| -------- | ---------------------------------------------------------------- |
| AGE      | Alzheimer’s risk increases with age                              |
| SEX      | Sex differences may influence disease prevalence and progression |
| EDUC     | Education relates to cognitive reserve                           |
| RAVLTFG  | Delayed verbal memory recall                                     |
| RAVLTIMM | Immediate verbal memory recall                                   |
| TRABSCOR | Executive function and processing speed                          |

These variables were intentionally chosen because they are biologically meaningful without directly encoding diagnosis.

---

## Removal of Semantically Leaky Features

Several highly diagnostic cognitive variables were intentionally removed:

* MMSCORE
* MOCA
* CDRSB
* FAQTOTAL

These variables produced near-perfect classification performance because they strongly overlap with the diagnostic criteria used to define Alzheimer’s disease itself.

Including them would create semantic leakage, where the model effectively memorizes diagnostic definitions instead of learning meaningful multimodal disease patterns.

Removing these variables created a more scientifically realistic and clinically defensible modeling framework.

---

## MRI Modality

The MRI modality captures structural neurodegeneration within the brain.

MRI features were extracted from the UCSFFSX6 FreeSurfer dataset.

These biomarkers describe:

* cortical thickness
* volumetric brain measurements
* ventricular enlargement
* regional atrophy patterns

Alzheimer’s disease is associated with structural degeneration in regions involved in:

* memory
* executive function
* cognition
* hippocampal processing

MRI therefore provides complementary anatomical information that differs fundamentally from behavioral cognitive testing.

---

# Technical Pipeline

## Step 1 — Convert ADNI .rda Files into CSV

The original ADNI datasets were distributed in R `.rda` format.

The datasets were converted into CSV format using R.

### Clinical Dataset Conversion

```r
setwd("/Users/suvankapoor/Documents/Multimodal Alzheimer's Research")

obj_name <- load("ADNIMERGE2/data/ADSL.rda")

df <- get(obj_name)

write.csv(df, "ADSL.csv", row.names = FALSE)
```

### MRI Dataset Conversion

```r
setwd("/Users/suvankapoor/Documents/Multimodal Alzheimer's Research")

obj_name <- load("ADNIMERGE2/data/UCSFFSX6.rda")

df <- get(obj_name)

write.csv(df, "UCSFFSX6.csv", row.names = FALSE)
```

---

## Step 2 — Python Preprocessing Pipeline

The preprocessing workflow included:

* diagnosis filtering
* patient-level merging
* label encoding
* MRI scan deduplication
* feature engineering
* missing value handling
* feature filtering
* scaling
* dimensionality reduction

---

## MRI Feature Engineering

The MRI dataset originally contained hundreds of structural variables.

To reduce overfitting:

* metadata variables were removed
* highly missing biomarkers were filtered out
* only numeric MRI biomarkers were retained
* variance filtering selected the top 30 MRI biomarkers

This reduced the MRI feature space to the most informative structural predictors.

---

## Missing Value Handling

Missing MRI values were handled using median imputation.

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
```

Median imputation was chosen because:

* it preserves significantly more patients
* it is robust to outliers
* it avoids major sample loss from row deletion

---

## Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

Stratified splitting ensured class balance between:

* cognitively normal patients
* dementia patients

---

## Feature Scaling

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Scaling stabilized model training and prevented variables with larger numeric ranges from dominating optimization.

---

# Machine Learning Models

Three machine learning models were evaluated:

| Model               | Purpose                           |
| ------------------- | --------------------------------- |
| Logistic Regression | Interpretable linear baseline     |
| Random Forest       | Nonlinear ensemble baseline       |
| Extra Trees         | High-variance ensemble comparison |

Logistic Regression ultimately emerged as the strongest and most stable model.

---

# Initial Model Comparison

## Baseline Model Performance

| Model               | ROC-AUC |
| ------------------- | ------- |
| Logistic Regression | Highest |
| Random Forest       | Strong  |
| Extra Trees         | Strong  |

Logistic Regression achieved the best balance of:

* accuracy
* stability
* interpretability
* robustness

---

# Visualization: Class Distribution

<img width="540" height="393" alt="image" src="https://github.com/user-attachments/assets/0732d553-5d01-418e-b6fa-2390283bc856" />

## What This Visualization Shows

This bar chart shows the number of patients in each diagnostic category:

* Diagnosis 0 = Cognitively Normal (CN)
* Diagnosis 1 = Dementia (DEM)

The dataset contains substantially more cognitively normal patients than dementia patients.

This imbalance is important because machine learning models can become biased toward the majority class. Because of this, evaluation metrics such as:

* ROC-AUC
* Recall
* Sensitivity
* F1-score

are more informative than accuracy alone.

The class imbalance also explains why stratified train/test splitting and balanced class weighting were used throughout the project.

---

# Visualization: Model Comparison

<img width="691" height="504" alt="image" src="https://github.com/user-attachments/assets/07e6676b-75bc-4c11-a09a-de9b2cd3d6e8" />

## What This Visualization Shows

This figure compares ROC-AUC performance across all evaluated machine learning models.

Key findings:

* Logistic Regression achieved the strongest overall performance
* Extra Trees also performed strongly
* Random Forest produced slightly lower performance

The results suggest that the Alzheimer’s classification problem was largely linearly separable after multimodal feature integration and preprocessing.

Because Logistic Regression also provides:

* strong interpretability
* stability
* lower overfitting risk

it was selected as the primary model for robustness experiments.

---

# Visualization: ROC Curves

<img width="613" height="624" alt="image" src="https://github.com/user-attachments/assets/943fa584-2639-4d34-93f5-02d03cf4413c" />

## What This Visualization Shows

ROC curves visualize how effectively each model separates:

* cognitively normal patients
* dementia patients

across all classification thresholds.

The x-axis represents:

* False Positive Rate

The y-axis represents:

* True Positive Rate (Sensitivity)

The dashed red line represents random guessing.

The closer a curve approaches the upper-left corner, the stronger the classifier.

The Logistic Regression model achieved the strongest ROC-AUC performance:

```text
ROC-AUC = 0.977
```

indicating extremely strong discriminative capability.

---

# Missing Clinical Modality Experiment

## Motivation

Real-world healthcare systems frequently contain incomplete clinical records.

To simulate this deployment scenario:

* all clinical features were removed from the test set
* MRI biomarkers remained available

This experiment evaluated how dependent multimodal systems become on clinical information.

---

## Missing-Modality Simulation

```python
X_test_missing[clinical_features] = 0
```

This simulated a realistic clinical scenario where:

* MRI scans exist
* cognitive testing is incomplete or unavailable

---

# Missing Clinical Modality Results

| Condition                 | ROC-AUC |
| ------------------------- | ------- |
| Full Multimodal Data      | 0.977   |
| Missing Clinical Modality | 0.771   |

## Interpretation

The baseline multimodal model experienced a major decline in performance when clinical information disappeared.

Performance degradation:

```text
0.205 ROC-AUC reduction
```

This demonstrates that standard multimodal systems can become highly dependent on clinical features.

---

# Robustness Training Through Modality Dropout

## Motivation

To improve resilience to missing data, modality dropout training was introduced.

During training:

* clinical variables were randomly removed for subsets of patients
* the model was forced to learn from MRI information alone when necessary

This prevented over-reliance on a single modality.

---

## Modality Dropout Implementation

```python
X_train_robust.loc[
    mask,
    clinical_features
] = 0
```

This simulates real-world clinical uncertainty during training.

---

# Cross-Validated Robustness Results

## 5-Fold Stratified Cross-Validation

To ensure statistical reliability:

* 5-fold stratified cross-validation was performed
* results were averaged across folds
* mean ± standard deviation values were reported

---

## Final Cross-Validated Results

| Model    | Mean Full AUC | Mean Missing AUC | Mean Performance Drop |
| -------- | ------------- | ---------------- | --------------------- |
| Baseline | 0.980 ± 0.007 | 0.840 ± 0.034    | 0.140 ± 0.028         |
| Robust   | 0.977 ± 0.008 | 0.863 ± 0.042    | 0.114 ± 0.037         |

---

# Biological and Clinical Interpretation of Results

## Key Finding

The robustness-trained model maintained nearly identical full-data performance while substantially improving resilience under missing clinical information.

This finding is clinically important because:

* healthcare records are often incomplete
* missing cognitive testing is common
* MRI imaging may still be available when clinical testing is absent
* deployable healthcare AI systems must tolerate imperfect data

The robust model reduced performance degradation from:

```text
0.140 → 0.114
```

representing a meaningful improvement in resilience.

---

# Missingness Robustness Experiments

## Motivation

The initial missing-modality experiment removed the entire clinical modality.

However, real-world healthcare missingness often occurs gradually.

Additional experiments therefore simulated increasing levels of missing clinical information.

---

## Missingness Levels Tested

* 10%
* 30%
* 50%
* 70%

Clinical variables were randomly masked to simulate Missing Completely At Random (MCAR) conditions.

---

# Missingness Experiment Results

| Missingness Level | Baseline AUC | Robust AUC | Baseline Performance Drop | Robust Performance Drop |
| ----------------- | ------------ | ---------- | ------------------------- | ----------------------- |
| 10%               | 0.919        | 0.987      | 0.058                     | 0.002                   |
| 30%               | 0.879        | 0.988      | 0.097                     | 0.001                   |
| 50%               | 0.820        | 0.980      | 0.156                     | 0.009                   |
| 70%               | 0.816        | 0.948      | 0.161                     | 0.041                   |

---

# Interpretation of Missingness Experiments

These experiments demonstrated:

* baseline multimodal models degrade steadily as missingness increases
* robustness-trained models remain substantially more stable
* modality dropout training distributes predictive importance more effectively across modalities

The robust model maintained strong discriminative performance even under severe missingness conditions.

---

# Visualization: Cross-Validated Robustness Performance

<img width="768" height="499" alt="image" src="https://github.com/user-attachments/assets/a525cb4d-cb6c-4434-878d-528e3005145e" />

## What This Visualization Shows

This figure compares:

* baseline multimodal performance
* robustness-trained multimodal performance
* performance under missing clinical data

Key findings:

* both models achieved nearly identical full-data performance
* the robust model maintained substantially stronger performance when clinical information disappeared
* modality dropout training reduced performance degradation under missingness

This demonstrates that robustness training improved resilience without sacrificing baseline predictive capability.

---

# Visualization: Missingness Robustness Curve

<img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/ee7ec3d2-67cb-406c-ab9f-0e3d677255d0" />

## What This Visualization Shows

This figure evaluates how model performance changes as increasing amounts of clinical data become missing.

The x-axis represents:

* percentage of missing clinical information

The y-axis represents:

* ROC-AUC performance

Key findings:

* the baseline model degrades steadily as missingness increases
* the robustness-trained model remains substantially more stable
* the robustness gap widens as missingness becomes more severe

This demonstrates that modality dropout training improves resilience under increasingly incomplete healthcare data conditions.

---

# Additional Baseline Experiments

## Motivation

Additional baselines were evaluated to determine:

* how predictive each modality is independently
* whether multimodal learning provides meaningful benefit

---

# Modality Comparison Results

| Model Type    | ROC-AUC | Feature Count |
| ------------- | ------- | ------------- |
| Clinical Only | 0.969   | 6             |
| MRI Only      | 0.825   | 30            |
| Multimodal    | 0.977   | 36            |

---

# Interpretation of Modality Results

## Clinical-Only Model

Clinical variables alone produced very strong predictive performance.

This suggests:

* cognitive testing contains highly informative behavioral disease signals
* memory impairment strongly differentiates dementia patients

---

## MRI-Only Model

MRI biomarkers alone achieved meaningful but weaker predictive performance.

This indicates:

* structural neurodegeneration remains highly informative
* MRI captures complementary anatomical information
* behavioral cognitive testing remains more directly diagnostic

---

## Multimodal Model

Combining MRI and clinical information produced the strongest overall performance.

This demonstrates:

* complementary integration of behavioral and anatomical disease signals
* multimodal learning improves overall discriminative capability

---

# Healthcare Evaluation Metrics

## Final Multimodal Performance Metrics

| Metric      | Value |
| ----------- | ----- |
| Accuracy    | 0.952 |
| Precision   | 0.786 |
| Recall      | 0.917 |
| F1-Score    | 0.846 |
| Sensitivity | 0.917 |
| Specificity | 0.958 |
| ROC-AUC     | 0.977 |

---

# Clinical Interpretation of Metrics

## Accuracy — 95.2%

The model correctly classified approximately 95% of patients overall.

---

## Recall / Sensitivity — 91.7%

Sensitivity measures:

> how effectively dementia patients are identified.

This is especially important clinically because missed dementia cases may delay diagnosis and treatment.

---

## Specificity — 95.8%

Specificity measures:

> how effectively cognitively normal patients are correctly identified.

High specificity reduces false dementia diagnoses.

---

## Precision — 78.6%

Precision measures:

> how many predicted dementia classifications were actually correct.

---

## F1-Score — 84.6%

The F1-score balances:

* precision
* recall

providing a robust summary metric under class imbalance.

---

## ROC-AUC — 0.977

ROC-AUC measures:

> the model’s ability to separate cognitively normal and dementia patients across all classification thresholds.

An ROC-AUC of 0.977 represents extremely strong discriminative capability.

---

# Confusion Matrix Interpretation

| Outcome         | Count |
| --------------- | ----- |
| True Negatives  | 68    |
| False Positives | 3     |
| False Negatives | 1     |
| True Positives  | 11    |

---

# Clinical Interpretation of Confusion Matrix

## True Positives — 11

The model correctly identified 11 dementia patients.

---

## False Negatives — 1

Only one dementia patient was missed.

This is clinically important because minimizing missed diagnoses is critical in healthcare AI systems.

---

## False Positives — 3

Only three cognitively normal patients were incorrectly classified as dementia.

---

# Visualization: Clinical Feature Correlation Heatmap

<img width="776" height="682" alt="image" src="https://github.com/user-attachments/assets/784d28ef-df85-4055-b47d-81c889b374ff" />

## What This Visualization Shows

This heatmap visualizes correlations between clinical variables.

Positive correlations indicate:

* variables that increase together

Negative correlations indicate:

* variables that move in opposite directions

Important observations include:

* delayed verbal recall and immediate verbal recall are moderately related
* trail-making performance exhibits inverse relationships with memory variables
* age demonstrates relationships with cognitive performance decline

This visualization helps validate that the clinical modality contains biologically meaningful cognitive relationships relevant to Alzheimer’s disease.

---

# Visualization: Confusion Matrix

<img width="435" height="393" alt="image" src="https://github.com/user-attachments/assets/90c9534d-9ce4-434e-a6de-6e6fcdc1e036" />

## What This Visualization Shows

The confusion matrix summarizes classification outcomes.

Rows represent:

* actual diagnoses

Columns represent:

* predicted diagnoses

Key findings:

* 68 cognitively normal patients were correctly identified
* 11 dementia patients were correctly identified
* only 1 dementia patient was missed
* only 3 cognitively normal patients were incorrectly classified as dementia

This demonstrates strong clinical sensitivity and specificity.

The very low false negative count is especially important because missed dementia diagnoses can delay treatment and intervention.

---

# Modality Dropout Ablation Study

## Motivation

The ablation study investigated:

> how different modality dropout rates influence robustness.

This experiment tested whether increasing dropout improves resilience to missing data.

---

# Dropout Levels Tested

* 0%
* 10%
* 30%
* 50%

---

# Ablation Study Results

| Dropout Rate | Full AUC | Missing AUC | Performance Drop |
| ------------ | -------- | ----------- | ---------------- |
| 0%           | 0.977    | 0.771       | 0.205            |
| 10%          | 0.977    | 0.817       | 0.160            |
| 30%          | 0.975    | 0.792       | 0.183            |
| 50%          | 0.969    | 0.827       | 0.142            |

---

# Interpretation of Ablation Study

These results demonstrated:

* no dropout creates fragile multimodal systems
* increasing modality dropout improves robustness
* higher dropout introduces a tradeoff between:

  * full-data optimization
  * missing-data resilience

The 50% dropout model produced the strongest robustness under missing clinical information.

---

# Visualization: Clinical vs MRI vs Multimodal Performance

<img width="613" height="470" alt="image" src="https://github.com/user-attachments/assets/ac27973c-8ce7-4aa6-9bdb-ed715850e039" />

## What This Visualization Shows

This figure compares the predictive strength of:

* clinical features alone
* MRI biomarkers alone
* multimodal integration

Key findings:

* clinical features were highly predictive
* MRI biomarkers alone remained meaningfully predictive
* combining both modalities produced the strongest overall performance

This demonstrates that multimodal learning provides complementary biological information.

---

# Visualization: Effect of Modality Dropout on Robustness

<img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/abe4bba7-6199-4dd0-bab6-0b63ea0de61b" />

## What This Visualization Shows

This figure evaluates how increasing modality dropout during training affects robustness under missing clinical data.

Key findings:

* no dropout produced the weakest robustness
* moderate dropout substantially improved resilience
* robustness improved as the model learned to rely less heavily on clinical variables

The strongest missing-modality performance occurred at higher dropout levels.

---

# Visualization: Effect of Dropout on Performance Degradation

<img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/41810060-b992-43dd-adff-2054ee721952" />

## What This Visualization Shows

This figure visualizes:

* how much performance is lost after clinical information disappears

Lower values indicate:

* stronger robustness
* less dependence on clinical variables

Key findings:

* no dropout produced the largest degradation
* modality dropout substantially reduced degradation
* robustness training distributed predictive importance more effectively across modalities

This figure demonstrates the central contribution of the project:

> modality dropout training improves resilience to incomplete healthcare data.

---

# Key Contributions

This project successfully developed:

* a multimodal Alzheimer’s disease classification framework
* a missing-modality evaluation pipeline
* a modality dropout robustness-training strategy
* cross-validated robustness experiments
* clinically interpretable multimodal analysis

---

# Main Scientific Findings

## 1. Multimodal Systems Are Vulnerable to Missing Clinical Information

Standard multimodal classifiers experienced substantial degradation when clinical variables disappeared.

---

## 2. Modality Dropout Significantly Improves Robustness

Robustness-trained models maintained substantially stronger performance under missing data.

---

## 3. MRI Biomarkers Provide Complementary Anatomical Information

MRI-only models remained predictive, demonstrating that structural neurodegeneration contains meaningful disease signal.

---

## 4. Robustness and Peak Accuracy Form a Tradeoff

Increasing dropout improves resilience but may slightly reduce optimal full-data performance.

---

# Clinical Importance

This project shifts the focus from:

> maximizing ideal-condition accuracy

to:

> designing clinically deployable healthcare AI systems.

Real healthcare environments contain:

* incomplete patient records
* inconsistent testing
* partial medical histories
* missing modalities

A model that remains stable under these conditions may ultimately be more clinically valuable than a model optimized only for ideal datasets.

---

# Limitations

Several limitations remain:

* ADNI may not fully represent real-world patient populations
* the study focused only on binary CN vs DEM classification
* synthetic missingness may not perfectly reflect clinical workflows
* external validation datasets were not yet incorporated
* MRI biomarkers were reduced using variance filtering rather than domain-specific selection

---

# Future Work

Potential future directions include:

* external validation on OASIS or NACC datasets
* longitudinal disease progression modeling
* advanced multimodal fusion architectures
* transformer-based multimodal systems
* clinically realistic Missing At Random (MAR) simulations
* fairness and demographic robustness analysis

---

# Technologies Used

## Languages

* Python
* R

## Machine Learning Libraries

* scikit-learn
* pandas
* numpy
* matplotlib
* seaborn

## Models

* Logistic Regression
* Random Forest
* Extra Trees

## Statistical Techniques

* Stratified Cross-Validation
* Median Imputation
* Standardization
* Variance Filtering
* ROC-AUC Evaluation
* Modality Dropout Training

---

# Repository Structure

```text
/data
/notebooks
/results
/figures
README.md
requirements.txt
```

---

# Final Conclusion

This project demonstrates that multimodal Alzheimer’s disease classifiers can achieve strong predictive performance while simultaneously exposing a major translational challenge:

> multimodal healthcare AI systems become vulnerable when patient information is incomplete.

Through modality dropout training, this work showed that:

* robustness to missing clinical information can be substantially improved
* resilience can increase without major loss in baseline accuracy
* clinically deployable machine learning systems should prioritize robustness in addition to predictive performance

The final framework therefore contributes not only to Alzheimer’s disease classification, but also to the broader problem of robust healthcare AI deployment under incomplete real-world data conditions.

