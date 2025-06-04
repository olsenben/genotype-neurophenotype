import mne
from mne.preprocessing import find_bad_channels_maxwell, ICA
from mne_icalabel import label_components
import pandas as pd
import os

#folder where data is stored
base_path = "E:/neuro_data/ds004796"

for i in range(1, 81):
    
    #STEP 1: Load Data
    subject = f"sub-{i:02d}"
    print(f"Processing folder: {subject}")
    file_prefix = (base_path, subject, "eeg")
    vhdr_file = os.path.join(file_prefix, f"{subject}_task-rest_eeg.vhdr")
    tsv_file = os.path.join(file_prefix, f"{subject}_task-rest_events.tsv")

    #instantiate
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
    
    #load tsv event data
    events_df = pd.read_csv(tsv_file, sep='\t')

    #create annotations object
    annotations = mne.Annotations(
        onset=events_df['onset'].values,
        duration=events_df['duration'].values,
        description=events_df['trial_type'].values
    )

    #set annotations
    raw.set_annotations(annotations)

    #STEP 2: Filter 1-100 Hz
    raw.filter(1.,100., fir_design='firwin')

    #STEP 3: Downsample
    raw_downsampled = raw.copy().resample(sfreq=250)

    #STEP 4: Set Common Average Reference (CAR)
    raw_downsampled.set_eeg_reference('average')

    #STEP 5: Channel Renaming, setting Montage, Marking bad channels
    raw_downsampled.rename_channels(lambda name: name.strip('.'))

#     - Split events (eyes open vs eyes close)
# - Run ICA 
# - Use ICLabel or manual review to find artifact components
# - Apply ICA to remove artifacts
# - Interpolate bad channels 
# - Epoch (10s)




