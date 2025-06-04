import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import mne
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(dataloader, desc="Training Batches...", leave=False): 
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels.unsqueeze(1)) #label shape mismatch
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0) #sum of loss over batch

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating...", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs,  labels.unsqueeze(1))
            running_loss += loss.item() * inputs.size(0)

            #binary classification, threshold outputs at 0.5
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)

    epochs_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return epochs_loss, accuracy

def compute_train_mean_std(dataset):
    all_data = []
    for x, _ in dataset:
        all_data.append(x)
    all_data = torch.stack(all_data)  #shape: (total_samples, channels, time)
    mean = all_data.mean(dim=(0, 2))  #mean across all samples and time, per channel
    std = all_data.std(dim=(0, 2))    #std across all samples and time, per channel
    
    return mean, std

     
def create_sampler(labels_tensor):
    class_counts = torch.bincount(labels_tensor.long())
    weights = 1. /class_counts.float()
    sample_weights = weights[labels_tensor.long()]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

class NormalizeEEG:
    def __init__(self, mean, std, eps=1e-6):
        """
        mean, std: torch tensors of shape (n_channels,)
        """
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, x):
        # x: (n_channels, n_times)
        return (x - self.mean[:, None]) / (self.std[:, None] + self.eps)
    
class AddGaussianNoise:
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class EEGEpochDataset(Dataset): 
    """class to hold subject risk labels"""
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

class EEGCNNClassifier(nn.Module):
    """parameters selected via: 
    
    Exploring the Effectiveness of Machine
    Learning and Deep Learning Techniques
    for EEG Signal Classification in
    Neurological Disorders
    SOUHAILA KHALFALLAH1,2, WILLIAM PEUCH3,
    (SENIOR MEMBER, IEEE), MEHDI TLIJA4,
    AND KAIS BOUALLEGUE.5
    
    """

    def __init__(self, in_channels=127, input_length=2500):
        super(EEGCNNClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 5, kernel_size=3),               # (B, 5, 2498)
            nn.BatchNorm1d(5),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),                  # (B, 5, 1249)

            nn.Conv1d(5, 5, kernel_size=3),                         # (B, 5, 1247)
            nn.BatchNorm1d(5),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),                  # (B, 5, 623)

            nn.Conv1d(5, 5, kernel_size=3),                         # (B, 5, 621)
            nn.BatchNorm1d(5),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.AvgPool1d(kernel_size=2, stride=2),                  # (B, 5, 310)

            # nn.Conv1d(5, 5, kernel_size=3),                         # (B, 5, 308)
            # nn.BatchNorm1d(5),
            # nn.LeakyReLU(),
            # nn.Dropout(0.2),
            # nn.AvgPool1d(kernel_size=2, stride=2),                  # (B, 5, 154)

            nn.Conv1d(5, 5, kernel_size=3),                         # (B, 5, 152)
            nn.LeakyReLU(),

            nn.AdaptiveAvgPool1d(output_size=1),                   # (B, 5, 1)
            nn.Flatten(),                                          # (B, 5)
            nn.Dropout(0.6),
            nn.Linear(5, 1),                                       # (B, 1)
            #nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    
if __name__=="__main__":
    
    data_dir = os.path.join('E:', 'neuro_data', 'processed', 'derivatives')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    label_dir = os.path.join(current_dir,'..', 'data', 'sub_risk_labels.json')


    with open(label_dir, 'r') as f:
        label_dict = json.load(f)
    
    #get all subjects available in data directory 
    all_subjects = [s for s in os.listdir(data_dir) if s in label_dict.keys()]
    labels = [label_dict[sub] for sub in all_subjects]

    #Train/Val Split
    train_ids, val_ids = train_test_split(all_subjects, test_size=0.2, stratify=labels, random_state=42)

    #temporary dataset to compute normalization stats
    temp_train_dataset  = EEGEpochDataset(data_dir, label_dict, subject_ids=train_ids)
    mean, std = compute_train_mean_std(temp_train_dataset)
    normalize = NormalizeEEG(mean, std)
    noise = AddGaussianNoise(std=.02)
    transform = Compose([normalize, noise])

    #create train/val datasets with transform
    train_dataset = EEGEpochDataset(data_dir, label_dict, subject_ids=train_ids, transform=transform)
    val_dataset = EEGEpochDataset(data_dir, label_dict, subject_ids=val_ids, transform=normalize)

    #weight samples 
    y_train_tensor = torch.tensor(train_dataset.labels)
    train_sampler = create_sampler(y_train_tensor)

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    #enable gpu support if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    model = EEGCNNClassifier()
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss() #binary classification with logits output
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    num_epochs = 20

    #training will stop if model begins to overfit
    best_val_loss = float('inf')
    patience = 5              # number of epochs to wait
    patience_counter = 0

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in tqdm(range(num_epochs), desc="Training Epochs..."):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - ")
        print(f"Train Loss: {train_loss:.4f} - ")
        print(f"Val Loss: {val_loss:.4f} - ")
        print(f"Val Acc: {val_acc:.4f}")

            # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            #Optional: Save the best model
            model_filepath = os.path.join(current_dir,'..', 'models')
            os.makedirs(model_filepath, exist_ok=True)
            model_filename = os.path.join(model_filepath, f'best_cnn_model.pt')
            torch.save(model.state_dict(), model_filename)

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    
    #Plot Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()

    #Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Validation Accuracy", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()