# EEG Dataset - PEARL-Neuro Database: EEG, fMRI, health and lifestyle data of middle-aged people at risk of dementia

## Overview

Full cohort: 192 healthy middle-aged (50-63) individuals, balanced female and male ratio.

### Genetic data (N = 192):
- Apolipoprotein E (APOE)
- Phosphatidylinositol binding clathrin assembly protein (PICALM)
- Basic demographic and health data
- Psychometric data (memory, intelligence, mood, personality, stress coping strategies)
- Cohort subgroup: 79 healthy middle-aged (50-63) individuals, balanced female and male ratio.

### Neuroimaging data:
- Functional data — electroencefalography (EEG) and functional magnetic resonance imaging (fMRI):
- Resting-state protocol (with two conditions: eyes open and eyes closed)
- Cognitive tasks: multi-source interference task (MSIT) and Sternberg’s memory task
- Blood tests data (blood count, lipid profile, HSV virus)

##  Download Instructions

This data is managed with version control and lightwieght access via Datalad. Do you not have to download the entire dataset (approx 240gb). 

# Clone the dataset (do not download files yet)
~~~ 
# Clone the dataset (do not download large files yet)
datalad install https://github.com/OpenNeuroDatasets/ds004796.git 

# Navigate to the dataset
cd file_path/ds004796

# Use a custom script to pull EEG files (eeg + vhdr + vmrk + tsv) per subject (see scripts)
python scripts/fetch_eeg_data.py
~~~

##  Notes
- No preprocessing at this stage
- Data retains original format
- EEG events are stored in .vmrk and .vhdr files

##  Further Notes
- This dataset is publically available via OPENEURO
- Please cite dataset for any publications

##  Resources
- [Datalad](https://handbook.datalad.org/en/latest/usecases/openneuro.html)
- [OpenNeuro](https://openneuro.org/datasets/ds004796/versions/1.0.9)
- [Exploring the Effectiveness of Machine Learning and Deep Learning Techniques for EEG Signal Classification in Neurological Disorders](https://www.researchgate.net/publication/388255558_Exploring_the_Effectiveness_of_Machine_Learning_and_Deep_Learning_Techniques_for_EEG_Signal_Classification_in_Neurological_Disorders)


