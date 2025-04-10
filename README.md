# Genotype-Neurophenotype Modeling & Biomarker Discovery

## Overview

This project investigates how genetic risk factors influence neurophysiological and cognitive phenotypes in healthy middle-aged peoples. The goal is to identify early biomarkers of neurodegenerative disease risk using multimodal data (genetics, EEG, fMRI, and psychometrics). A follow-up phase expands to clinical EEG data for disease classification.

- **Primary Goal**: Predict neurocognitive features and extract biomarkers from genotype and neuroimaging data.
- **Expansion Goal**: Apply findings to clinical data for diagnostic use cases (e.g., Alzheimer's vs. FTD classification).
- **Technologies**: 

---

## Phase 1: Genotype-Neurophenotype Modeling (Healthy Aging)

### Dataset: PEARL-Neuro (N=192)
- Genetic markers: **APOE**, **PICALM**
- Neuroimaging: **EEG (resting-state, cognitive tasks)**, **fMRI**
- Cognitive tasks: **Sternberg Memory**, **MSIT**
- Psychometrics: memory, intelligence, mood, personality
- Health data: blood tests, demographics

### Objectives
- Model the relationship between **genotype** and **neurophenotype**
- Extract **EEG/fMRI biomarkers** associated with risk alleles
- Identify possible predictors of cognitive decline
- Apply **feature attribution methods** (e.g., SHAP, GradCAM) for biological interpretability

---

## Phase 2: Clinical Expansion (Alzheimer’s & FTD EEG Classification)

### Dataset: AHEPA General Hospital EEG Dataset
- Subjects: 36 with Alzheimer’s, 23 with FTD, 29 healthy controls
- 10–20 EEG, 500Hz, >19 hours of recordings

### Objectives
- Build deep learning models to classify **Alzheimer’s**, **FTD**, and **controls**
- Validate if features from Phase 1 generalize to clinical data
- Use explainable AI to localize and understand neural changes in dementia
- Compare feature spaces between healthy aging and disease

---

## Methodology

- Preprocessing: MNE, fMRIPrep, artifact rejection, source localization
- Feature Engineering:
  - EEG: Mean, Standard Deviation, Maximum value, Minimum value, Variance, Skewnewss, Kurtosis, Shannon entropy, Pwelwich, Band Power 
  - For more info [read](https://www.researchgate.net/publication/388255558_Exploring_the_Effectiveness_of_Machine_Learning_and_Deep_Learning_Techniques_for_EEG_Signal_Classification_in_Neurological_Disorders)
  - fMRI connectivity, ICA components
- Modeling:
  - ML: Logistic Regression, SVMs, Random Forests
  - DL: CNNs for EEG spectrograms, Transformer variants
  - Interpretability: SHAP, GradCAM, PCA
- Evaluation:
  - Cross-validation, AUROC, F1 score, model robustness tests

---

## Data Sources

| Dataset | Description | Usage |
|--------|-------------|-------|
| A Polish Electroencephalography, Alzheimer’s Risk-genes, Lifestyle and Neuroimaging (PEARL-Neuro) Database| N=192, includes EEG/fMRI, genetics, psychometrics | Phase 1 – Biomarker Discovery | 
| AHEPA EEG | Alzheimer’s, FTD, Controls EEG recordings | Phase 2 – Clinical Classification | 

- https://www.nature.com/articles/s41597-024-03106-5
= https://www.medrxiv.org/content/10.1101/2024.08.05.24311520v1

---

## Future Directions

- Integrate additional omics data (e.g., transcriptomics)
- Longitudinal modeling with follow-up neurodegeneration studies
- Investigate gene-environment interactions (stress, mood vs. risk)
- Train contrastive/self-supervised models on EEG/fMRI for generalized neurofeature learning
