from nilearn.input_data import NiftiLabelsMasker
from nilearn import datasets, image
import numpy as np
import torch
import os
import json
from tqdm import tqdm
import traceback

"""
do NOT forget to fetch the data via fetch_fmri_data.py. 
I spent an hour trying to debug this before realizing I never downloaded the data.
"""
# === Config ===
bold_suffix = "_task-rest_dir-PA_bold.nii.gz"
data_dir = os.path.join('E:\\', 'neuro_data', 'ds004796')
label_file = os.path.join(os.getcwd(), 'data', 'sub_risk_labels.json')
output_dir = os.path.join('E:\\', 'neuro_data',  'processed', 'derivatives')

# === Load label dictionary ===
with open(label_file, 'r') as file:
    label_dict = json.load(file)
    
# === Load atlas ===
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=True)

# === error & subject log ===
error_log = {}
subjects = []

# === processing loop ===
for subject_id in tqdm(label_dict.keys(), desc="Extracting Avg fMRI featurs..."):
    bold_file = os.path.join(data_dir, f"{subject_id}","func",f"{subject_id}{bold_suffix}")

    if not os.path.exists(bold_file):
        print(f"\nMissing BOLD file for {subject_id}")
        continue
    
    try:
        bold_img = image.load_img(bold_file)
        time_series = masker.fit_transform(bold_img) #(n_timepoints, n_rois)
        mean_features = time_series.mean(axis=0) #(n_rois,)

        tensor_feat = torch.tensor(mean_features, dtype=torch.float32)
        torch.save(tensor_feat, os.path.join(output_dir, subject_id,f"{subject_id}_fmri.pt"))
        subjects.append(subject_id)

    except Exception as e: 
        print(f"Error Processing {subject_id}: {str(e)}")
        error_log[subject_id] = traceback.format_exc()


if error_log:
    with open(r"outputs\preprocessing_fmri_errors.txt", "w") as f:
        for subject, error_trace in error_log.items():
            f.write(f"-----{subject}-----\n")
            f.write(error_trace + "\n\n")
    print("Some files have failed. See 'preprocessing_errors' for details")

else:
    print("All files process successfully")

if subjects:
    with open(r'data\available_fmri_subjects.json', 'w') as j:
        json.dump(subjects, j)
    print("saved available fmri subjects")


