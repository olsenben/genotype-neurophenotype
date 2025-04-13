import subprocess
import os

"""
Run to fetch resting eeg data via datalab. change basepath to your directory.
Make sure you have datalab installed.

clone the dataset (do not download large files yet)
datalad install https://github.com/OpenNeuroDatasets/ds004796.git 

Navigate to the dataset
cd file_path/ds004796

Use to pull EEG files (eeg + vhdr + vmrk + tsv)

Do not rename or move files from this directory!
"""

base_path = "E:/neuro_data/ds004796"

for i in range(1, 81):
    subject = f"sub-{i:02d}"
    eeg_file = os.path.join(base_path, subject, "eeg", f"{subject}_task-rest_eeg.eeg")
    vhdr_file = os.path.join(base_path, subject, "eeg", f"{subject}_task-rest_eeg.vhdr")
    vmrk_file = os.path.join(base_path, subject, "eeg", f"{subject}_task-rest_eeg.vmrk")
    tsv_file = os.path.join(base_path, subject, "eeg", f"{subject}_task-rest_events.tsv")
    subprocess.run(["datalad", "get", eeg_file, vhdr_file, vmrk_file, tsv_file])