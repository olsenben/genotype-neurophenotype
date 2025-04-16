import mne
from mne import make_fixed_length_epochs
from mne.preprocessing import find_bad_channels_maxwell, ICA
from mne_icalabel import label_components
import pandas as pd
import os
import traceback

"""
script for preprocessing resting eeg data. 
if you only want to process eyes open vs eyes closed,
make sure you comment out the appropriate lines after 
the event splitting step (Step 8)
Warning: each ICA fitting can 5-8 minutes. Preprocessing this data will take awhile, go get a coffee or something.
"""

#folder where data is stored
base_path = "E:/neuro_data/ds004796"

error_log = {}

def get_segments(annotations):
    #decscriptions are "event_type"
    descs = [desc.strip() for desc in annotations.description]
    onsets = annotations.onset

    events_dict = {}

    for desc, onset in zip(descs, onsets):
        if desc not in events_dict:
            events_dict[desc] = []
        events_dict[desc].append(onset)

    #eyes open, prefer S 2 - S 10
    if "S  2" in events_dict:
        open_start = events_dict["S  2"][0]
    elif "S  1" in events_dict:
        open_start = sorted(events_dict["S  1"])[0]
    else:
        raise ValueError("No valid start for eyes-open (S 2 or S 1)")
    
    #find open_end
    if "S  10" in events_dict:
        open_end = events_dict["S  10"][0]
    else:
        #go to next event as fallback
        next_events = [t for t in onsets if t > open_start]
        open_end = next_events[0] if next_events else open_start + 240 #assuming 4 minutes based on observations

    #eyes closed, prefer S 4 - S 11
    if "S  4" in events_dict:
            closed_start = events_dict["S  4"][0]
    elif "S  1" in events_dict:
        closed_candidates = [t for t in events_dict["S  1"] if t > open_end]
        closed_start = closed_candidates[0] if closed_candidates else open_end + 1
    else:
        raise ValueError("No valid start for eyes-closed (S 4 or S 1)")
    
    #find closed_end
    if "S  11" in events_dict:
        closed_end = events_dict["S  11"][0]
    else:
        #go to next event as fallback
        next_events = [t for t in onsets if t > closed_start]
        closed_end = next_events[0] if next_events else open_start + 360 #assuming 6 minutes based on observations

    return (open_start, open_end), (closed_start, closed_end)

for i in range(1, 81):
    subject = f"sub-{i:02d}"
    try:
        #STEP 1: Load Data
        print(f"Processing folder: {subject}")
        file_prefix = os.path.join(base_path, subject, "eeg")
        vhdr_file = os.path.join(file_prefix, f"{subject}_task-rest_eeg.vhdr")
        tsv_file = os.path.join(file_prefix, f"{subject}_task-rest_events.tsv")

        raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
        
        events_df = pd.read_csv(tsv_file, sep='\t')

        #STEP 2: Filter 1-100 Hz
        raw.filter(1.,100., fir_design='firwin')

        #STEP 3: Downsample
        raw_downsampled = raw.copy().resample(sfreq=250)

        #STEP 4: Set CAR
        raw_downsampled.set_eeg_reference('average')

        #STEP 5: Rename Channels and set Montage
        raw_downsampled.rename_channels(lambda name: name.strip('.'))

        #SETP 6: Redimensioning 

        #STEP 7: Marking Bad Channels
        try:
            raw_downsampled.info['bads'],_ = find_bad_channels_maxwell(raw)
        except ValueError as e:
            print(e)

        #STEP 8: Set annotations and split Events
        annotations = mne.Annotations(
        onset=events_df['onset'].values,
        duration=events_df['duration'].values,
        description=events_df['event_type'].values
        )

        raw_downsampled.set_annotations(annotations)
        (open_start, open_end), (closed_start, closed_end) = get_segments(raw_downsampled.annotations)

        #raw_downsampled_open = raw.copy().crop(tmin=open_start, tmax=open_end)
        raw_downsampled_closed = raw.copy().crop(tmin=closed_start, tmax=closed_end)
        
        #STEP 9: Artifact Removal
        ica = ICA(n_components=.99,method='picard',fit_params=dict(ortho=False, extended=True), random_state=97)

        #ica.fit(raw_downsampled_open)
        ica.fit(raw_downsampled_closed)

        labels = label_components(raw_downsampled_closed, ica, method='iclabel')

        bad_labels = ['eye','heart','muscle']

        bad_idx = [i for i, label in enumerate(labels['labels']) if label in bad_labels]

        print(f"Removing ICA components: {bad_idx} - {', '.join(labels['labels'][i] for i in bad_idx)}")

        ica.exclude = bad_idx

        #raw_clean_open = ica.apply(raw_downsampled_open)
        raw_clean_closed = ica.apply(raw_downsampled_closed)

        #raw_clean_open.interpolate_bads()
        raw_clean_closed.interpolate_bads()

        #STEP 10: Filter 0.5-60 Hz
        iir_params = dict(order=5, ftype='butter', output='sos')
        
        raw_clean_closed.filter(l_freq=0.5, h_freq=None, method='iir', iir_params=iir_params)

        raw_clean_closed.filter(l_freq=None, h_freq=60, method='fir', fir_design='firwin')

        epochs_closed = make_fixed_length_epochs(raw_clean_closed, duration=10, preload=True)

        #StEP 11: Save to disk
        save_dir = os.path.join("E:/neuro_data/processed/derivatives",subject, "eeg")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{subject}_task-rest_preproc-epo.fif")
        
        epochs_closed.save(save_path, overwrite=True)
    
    except Exception as e:
        print(f"Error Processing {subject}: {str(e)}")
        error_log[subject] = traceback.format_exc()

#save error txt
if error_log:
    with open("preprocessing_errors.txt", "w") as f:
        for subject, error_trace in error_log.items():
            f.write(f"-----{subject}-----\n")
            f.write(error_trace + "\n\n")
    print("Some files have failed. See 'preprocessing_errors' for details")

else:
    print("All files process successfully")

    





