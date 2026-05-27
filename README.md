# The-Impact-of-Missing-Clinical-Data-on-Multimodal-Alzheimer-s-Disease-Classification

# Improving Robustness of Multimodal Alzheimer’s Disease Classification Under Missing Clinical Data

## Overview

This project investigates the robustness of multimodal machine learning systems for Alzheimer’s disease classification under conditions of incomplete clinical information. The central motivation comes from a major challenge in real-world healthcare systems: patient data is rarely complete. Clinical assessments may be unavailable, cognitive testing may be partially administered, or electronic medical records may contain missing fields. While many machine learning studies assume perfectly complete datasets, clinical deployment environments are far less controlled. The goal of this work was therefore not simply to build a high-performing Alzheimer’s classifier, but to evaluate how performance changes when one source of information becomes unavailable and whether models can be trained to become more resilient to this missingness.

The project used data from the Alzheimer’s Disease Neuroimaging Initiative (ADNI), one of the largest and most widely used public datasets for Alzheimer’s disease research. ADNI contains multimodal patient information including neuroimaging, cognitive testing, demographics, and longitudinal follow-up data. Two major sources of information were incorporated into the pipeline. The first consisted of clinical and cognitive variables obtained from the ADSL dataset. These included demographic variables such as age, sex, and years of education, as well as memory and cognitive performance measures including delayed verbal recall, immediate verbal recall, and trail-making performance scores. The second modality consisted of structural MRI biomarkers extracted from the UCSFFSX6 dataset, which contains FreeSurfer-derived neuroimaging measurements describing structural brain anatomy and atrophy patterns associated with neurodegeneration.

The workflow began in R because the original ADNI files were distributed in the .rda format, which is native to the R ecosystem. The clinical and MRI datasets were loaded into R, converted into CSV format, and then transferred into Python for all subsequent preprocessing and machine learning analysis. Once loaded into Python, the data pipeline focused on creating a clean, reproducible multimodal dataset suitable for machine learning experimentation. Patients were filtered to include only cognitively normal individuals (CN) and dementia patients (DEM), producing a binary classification task. Diagnosis labels were encoded numerically to support downstream machine learning models.

A substantial portion of the work involved carefully designing the feature set in a biologically meaningful and scientifically defensible way. Initially, highly diagnostic clinical variables such as MMSE, CDRSB, MOCA, and FAQ scores were included. However, exploratory analysis revealed that several of these features alone could nearly perfectly predict diagnosis. This created a semantic leakage problem in which the model was effectively learning diagnostic criteria rather than discovering meaningful multimodal disease patterns. These features were therefore removed from the final modeling pipeline. Instead, the final clinical modality focused on weaker but biologically relevant predictors including age, sex, education, delayed verbal memory recall, immediate verbal recall, and trail-making performance.

The MRI modality represented a fundamentally different source of information. While the clinical modality captures observable cognitive and behavioral performance, the MRI modality captures structural neurodegeneration within the brain itself. The FreeSurfer MRI features describe volumetric and structural measurements across numerous brain regions that are known to be affected in Alzheimer’s disease. Because the MRI dataset contained hundreds of imaging variables, dimensionality reduction was necessary to avoid overfitting. The top MRI biomarkers were therefore selected using variance-based filtering, allowing the final model to focus on the most informative structural features.

After cleaning and preprocessing, the clinical and MRI datasets were merged at the patient level using RID identifiers. Missing MRI values were handled through median imputation rather than row deletion in order to preserve patient counts and maintain statistical power. The final multimodal dataset combined clinical variables with structural MRI biomarkers, creating a biologically meaningful representation of both behavioral and anatomical disease manifestations.

Multiple machine learning models were evaluated, including Logistic Regression, Random Forest, and Extra Trees classifiers. The dataset was divided into stratified training and testing partitions, and feature scaling was applied to ensure stable model training. Initial experiments demonstrated that the multimodal models achieved strong classification performance, with Logistic Regression ultimately emerging as the strongest and most stable model. After removal of semantically leaky variables, the final baseline multimodal system achieved a realistic and highly credible classification performance.

The most important component of the project involved evaluating robustness under missing modalities. To simulate realistic clinical missingness, all clinical features were removed from the test dataset while MRI features remained available. Conceptually, this experiment represents a deployment scenario in which imaging information exists but clinical testing data is unavailable or incomplete. Under these conditions, the baseline model experienced a substantial decline in performance, demonstrating that multimodal systems can become highly dependent on clinical information.

The central contribution of the project came from the implementation of modality dropout training. During training, clinical features were intentionally removed for random subsets of patients. This forced the model to learn how to make predictions even when clinical information was unavailable. Biologically, this simulates the uncertainty and incompleteness that frequently occur in real clinical environments. Rather than overfitting to a single highly informative modality, the model learned to distribute predictive importance across both MRI and clinical information sources.

The robustness-trained model produced one of the most important findings of the project. While maintaining nearly identical full-data performance, the robust model performed substantially better when clinical data was missing. The degradation in predictive performance was significantly reduced compared to the baseline model, demonstrating that simple modality dropout strategies can meaningfully improve resilience to incomplete patient information.

From a biological and translational perspective, this finding is important because it shifts the focus away from simply maximizing classification accuracy and toward designing clinically deployable machine learning systems. In practice, healthcare AI systems must operate under conditions of imperfect and incomplete information. A model that performs slightly worse under ideal conditions but remains stable when data is missing may ultimately be far more useful in real-world medicine.

The project therefore evolved from a standard Alzheimer’s disease classification study into a broader investigation of robustness, multimodal learning, and resilience under clinical missingness. The final framework demonstrates that multimodal machine learning systems can be trained not only to classify disease effectively, but also to tolerate incomplete healthcare data in a substantially more reliable manner.

# Technical Pipeline

## Data Pipeline Overview

The overall technical workflow for the project was divided into several stages:

1. Convert ADNI `.rda` files into CSV format using R
2. Load and preprocess datasets in Python
3. Create biologically meaningful multimodal feature groups
4. Merge clinical and MRI data at the patient level
5. Handle missing values and feature filtering
6. Train and evaluate machine learning models
7. Simulate missing modalities
8. Train robustness-enhanced models using modality dropout

---

## R Data Conversion Pipeline

The ADNI datasets were originally distributed in the `.rda` format. Because these files are native to the R ecosystem, the first step of the project involved converting them into CSV files that could later be processed in Python.

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

## Python Preprocessing Pipeline

Once the datasets were converted into CSV format, Python was used for all downstream preprocessing and machine learning analysis.

The preprocessing pipeline included:

* Diagnosis filtering
* Label encoding
* Clinical feature engineering
* MRI feature filtering
* MRI scan deduplication
* Missing value handling
* Patient-level dataset merging
* MRI dimensionality reduction

---

## Clinical Feature Engineering

The final clinical modality consisted of:

* AGE
* SEX
* EDUC
* RAVLTFG
* RAVLTIMM
* TRABSCOR

The following highly diagnostic variables were intentionally removed:

* MMSCORE
* MOCA
* CDRSB
* FAQTOTAL

These variables were removed because they created semantic leakage and produced unrealistically perfect classification performance.

---

## MRI Feature Engineering

The MRI modality was derived from the UCSFFSX6 FreeSurfer dataset.

The MRI pipeline involved:

* Selecting the latest MRI scan per patient
* Removing metadata variables
* Filtering out highly missing MRI biomarkers
* Keeping only numeric structural MRI variables
* Selecting the top 30 MRI biomarkers using variance filtering

This reduced the MRI feature space from hundreds of variables to a smaller and more stable subset suitable for machine learning.

---

## Missing Value Handling

Missing MRI values were handled using median imputation:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
```

Median imputation was chosen because it preserved significantly more patients than complete-row deletion while remaining robust to outliers.

---

## Machine Learning Pipeline

Three machine learning models were evaluated:

* Logistic Regression
* Random Forest
* Extra Trees

The workflow consisted of:

* Stratified train/test splitting
* Standard feature scaling
* Model comparison using ROC-AUC
* Visualization of ROC curves and feature importance

### Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

### Feature Scaling

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Logistic Regression Model

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
```

---

## Missing Modality Experiment (Cell 4)

The first major experiment simulated missing clinical information.

During testing, all clinical variables were removed:

```python
X_test_missing[clinical_features] = 0
```

This simulated a real-world clinical scenario where:

* MRI data exists
* Clinical assessments are unavailable or incomplete

The purpose of this experiment was to measure how dependent the multimodal system was on the clinical modality.

The baseline model showed a substantial performance decline under this missingness condition.

---

## Robustness Training (Cell 5)

The final experiment introduced a robustness-training strategy known as modality dropout.

During training, clinical variables were randomly removed for subsets of patients:

```python
X_train_robust.loc[
    mask,
    clinical_features
] = 0
```

This forced the model to learn how to make predictions even when clinical information was unavailable.

The resulting robustness-trained model maintained nearly identical full-data performance while dramatically improving performance under missing-clinical conditions.

This became the central contribution of the project.

---

# Visualizations


### Figure 1 — Class Distribution


<img width="540" height="393" alt="c4b0fa50-989f-40ba-8d7b-241c60c2577d" src="https://github.com/user-attachments/assets/f6788b1f-8481-4eee-90e5-864642ae2429" />


Visualization of cognitively normal versus dementia patient counts.

### Figure 2 — Model Comparison


<img width="691" height="504" alt="bf5e8e98-4ced-4320-8cb3-33733ea93801" src="https://github.com/user-attachments/assets/f48c388a-cc02-4f9c-8051-bac6e8159b21" />


Comparison of Logistic Regression, Random Forest, and Extra Trees AUC performance.

### Figure 3 — ROC Curves


<img width="540" height="393" alt="c4b0fa50-989f-40ba-8d7b-241c60c2577d" src="https://github.com/user-attachments/assets/f6788b1f-8481-4eee-90e5-864642ae2429" />


Receiver operating characteristic curves for all evaluated models.

### Figure 4 — Feature Importance


<img width="435" height="393" alt="2da517ef-ff19-4ed5-9592-0b950e06e708" src="https://github.com/user-attachments/assets/9d282478-f533-439f-9a99-ca17e85e6f04" />


Visualization of the most influential MRI and clinical predictors.

### Figure 5 — Missing Modality Performance


<img width="776" height="682" alt="bfd1e70a-7050-4865-99f1-6ce300aa62f9" src="https://github.com/user-attachments/assets/498e56b3-58fe-483d-9be1-045f21a00d15" />


Comparison between full multimodal performance and missing-clinical-modality performance.

### Robustness Training Results

Comparison of baseline versus robustness-trained model performance under missing clinical data.

Original Full AUC:
0.9765258215962441

Original Missing-Modality AUC:
0.7711267605633803

Robust Model Full AUC:
0.9800469483568075

Robust Model Missing-Modality AUC:
0.8661971830985916

Original Performance Drop:
0.20539906103286387

Robust Performance Drop:
0.11384976525821588<img width="613" height="624" alt="fbe9e9d6-81a6-4021-9c8f-19eb1fff0575" src="https://github.com/user-attachments/assets/33d96e3f-2b01-462e-b29b-2ec4f7568deb" />


# Final Summary

This project successfully developed a multimodal machine learning framework for Alzheimer’s disease classification while investigating how missing clinical information affects predictive performance. By combining structural MRI biomarkers with clinical and cognitive features, the study demonstrated strong multimodal classification capability while simultaneously revealing the vulnerability of such systems to incomplete data.

Most importantly, the project showed that simple robustness strategies based on modality dropout training can substantially reduce performance degradation under missing clinical information. This finding highlights the importance of designing healthcare AI systems that are not only accurate under ideal conditions, but also resilient under the imperfect and incomplete data conditions that characterize real-world clinical environments.

This project successfully built:

* A multimodal Alzheimer’s disease classifier
* A modality robustness evaluation framework
* A robustness-training strategy using modality dropout

The final findings demonstrated that:

> Training models under simulated missingness substantially improves robustness to incomplete clinical data.

This creates a stronger foundation for real-world clinical machine learning deployment.
