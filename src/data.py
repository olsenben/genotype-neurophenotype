
import tqdm as tqdm
import os
import mne
import torch
from torch.utils.data import Dataset

class EEGEpochDataset(Dataset): 
    """class to hold subject risk labels and eeg epochs"""
    def __init__(self, data_dir, label_dict, subject_ids=None, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        self.subject_ids = set(subject_ids) if subject_ids is not None else None
        
        #load all files in data directory and convert to tensors 
        for dirpath, __, filenames in tqdm(os.walk(data_dir),desc="Loading EEG Data..."):
            for filename in filenames:
                if filename.lower().endswith('.fif'):
                    full_path = os.path.join(dirpath, filename)
                    subject_id = os.path.basename(full_path).split('_')[0]
                    
                    #skip missing data
                    if subject_id not in label_dict.keys():
                        continue
                    if self.subject_ids and subject_id not in self.subject_ids:
                        continue
                    
                    #add a pass through for errors
                    try:
                        label = label_dict[subject_id]
                        epochs = mne.read_epochs(full_path, preload=True)
                        data = epochs.get_data() #(n_epochs, n_channels, n_times)

                        for epoch in data:
                            tensor_epoch = torch.tensor(epoch, dtype=torch.float32)
                            if self.transform:
                                tensor_epoch = self.transform(tensor_epoch)
                            self.samples.append(tensor_epoch)
                            self.labels.append(torch.tensor(label, dtype=torch.float32))
                    except: 
                        print (f"{subject_id} could not be processed")
                        continue
                    

    def __len__(self):
        """how many subject .fif files found in directory?"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """retrieves sample, label tensors for given index in list of samples"""
        return self.samples[idx], self.labels[idx]


class EEGFMRIDataset(Dataset):
    """class to hold subject risk labels, eeg epochs, fmri data"""

    def __init__(self, data_dir, label_dict, subject_ids=None, transform=None):
        self.samples = []
        self.fmri_features = []
        self.labels = []
        self.transform = transform
        self.subject_ids = set(subject_ids) if subject_ids else None

        for dirpath, _, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.lower().endswith('.fif'):
                    subject_id = filename.split('_')[0]
                    if subject_id not in label_dict or (self.subject_ids and subject_id not in self.subject_ids):
                        continue

                    try:
                        # Load EEG
                        epochs = mne.read_epochs(os.path.join(dirpath, filename), preload=True)
                        data = epochs.get_data()
                        # Load precomputed fMRI features (same shape for all subjects)
                        fmri_feat = torch.load(os.path.join(data_dir, subject_id,f"{subject_id}_fmri.pt"))  # shape: (N_feat,)
                        label = label_dict[subject_id]

                        for epoch in data:
                            tensor_epoch = torch.tensor(epoch, dtype=torch.float32)
                            if self.transform:
                                tensor_epoch = self.transform(tensor_epoch)
                            self.samples.append(tensor_epoch)
                            self.fmri_features.append(fmri_feat)
                            self.labels.append(torch.tensor(label, dtype=torch.float32))

                    except Exception as e:
                        print(f"Skipping {subject_id} due to error: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.fmri_features[idx], self.labels[idx]
