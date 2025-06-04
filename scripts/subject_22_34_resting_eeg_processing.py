import mne
from mne import make_fixed_length_epochs
from mne.preprocessing import find_bad_channels_maxwell, ICA
from mne_icalabel import label_components
import pandas as pd
import os
import traceback

"""
There were some encoding errors, mainly subject 22 and 34 were missing some events. 
subject 22 was missing eyes open resting events, subject 34 the events were encoded wrong.
subject 55 is also missing rest data 
subject 69 is missing from the dataset.
Since its just those two I am fixing them manually.
"""
error_log = {}

#folder where data is stored
base_path = "E:/neuro_data/ds004796"

subjects = {
    'sub-22' : {
        'S 10' : 240.773,
        'S  1' : 253.537,
        'S 11' : 614.83
    },
    'sub-34' : {
        'S  1' : 246.55,
        'S 11' : 702.138
    }
}

for subject, data in subjects.items():
    subject = subject
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

        #(open_start, open_end), 
        (closed_start, closed_end) = subjects[subject]['S  1'], subjects[subject]['S 11']

        #raw_downsampled_open = raw_downsampled.copy().crop(tmin=open_start, tmax=open_end)
        raw_downsampled_closed = raw_downsampled.copy().crop(tmin=closed_start, tmax=closed_end)
        
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

