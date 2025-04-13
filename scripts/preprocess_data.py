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

    #instatiate
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

    #STEP 2: Channel Renaming and setting Montage

    #strip trailing dots from channel names
    raw.rename_channels(lambda name: name.strip('.'))

    #STEP 3: Redimensioning
    
    #STEP 4: Marking Bad Channels
    try:
        raw.info['bads'],_ = find_bad_channels_maxwell(raw)
    except ValueError as e:
        print(e)
    
    #STEP 5: Artifact Removal

    #bfore filtering, set common average reference for labeling
    raw.set_eeg_reference('average', projection=True)

    #apply projectionss for improved artifact removal
    raw.apply_proj()

    #apply filter for mne_icalabel which expects prefiltered data 1-100hz
    raw.filter(1.,100., fir_design='firwin')

    #create ICA object with picard from scikit-learn. default n_components=.99 which captures 99% of varience
    #between infomax, which is a more classic approach, and fastfit from scikit-learn, picard is the quickest
    #for ICLabel, you should select infomax or picard
    ica = ICA(n_components=.99,method='picard',fit_params=dict(ortho=False, extended=True), random_state=97)

    print(f"Fitting {subject} ICA")

    ica.fit(raw)

    #label components
    labels = label_components(raw, ica, method='iclabel')

    #components to be ignored
    bad_labels = ['eye','heart','muscle']

    #index for components to be ignored
    bad_idx = [i for i, label in enumerate(labels['labels']) if label in bad_labels]

    print(f"Removing ICA components: {bad_idx} - {', '.join(labels['labels'][i] for i in bad_idx)}")

    #remove components
    ica.exclude = bad_idx

    #apply ICA to raw daya
    raw_clean = ica.apply(raw)

    #using the cleared and fitted data, we can now interpolate the bad channels from earlier (these were ignored automatically during ICA)
    raw_clean.interpolate_bads()

    #STEP 6: Filter

    #high pass IIR filter to remove drift < 0.5 Hz. I guess its already filtered at 1 Hz but I wrote this before I switched methods so I'll leave it
    iir_params = dict(order=5, ftype='butter', output='sos')
    raw_clean.filter(l_freq=0.5, h_freq=None, method='iir', iir_params=iir_params)

    #low pass FIR filter at 58 Hz
    raw_clean.filter(l_freq=None, h_freq=60, method='fir', fir_design='firwin')

    #Step 7: Downsample
    raw_downsampled = raw_clean.copy().resample(sfreq=250)

    #Step 8: Epoch

    #Step 9: Save Prepocessed Data




