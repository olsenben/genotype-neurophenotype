import subprocess
import os

base_path = "E:/neuro_data/ds004796"
for i in range(1, 81):
    subject = f"sub-{i:02d}"
    eeg_file = os.path.join(base_path, subject, "eeg", f"{subject}_task-rest_eeg.eeg")
    vhdr_file = os.path.join(base_path, subject, "eeg", f"{subject}_task-rest_eeg.vhdr")
    vmrk_file = os.path.join(base_path, subject, "eeg", f"{subject}_task-rest_eeg.vmrk")
    tsv_file = os.path.join(base_path, subject, "eeg", f"{subject}_task-rest_events.tsv")
    subprocess.run(["datalad", "get", eeg_file, vhdr_file, vmrk_file, tsv_file])