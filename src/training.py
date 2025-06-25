import torch
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler


def train_one_eeg_epoch(model, dataloader, loss_fn, optimizer, device):
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

def evaluate_eeg(model, dataloader, loss_fn, device):
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

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """Modified training epoch for eeg and fmri data"""
    model.train()
    running_loss = 0.0

    for eeg_inputs, fmri_inputs, labels in tqdm(dataloader, desc="Training Batches...", leave=False): 
        eeg_inputs, fmri_inputs, labels = eeg_inputs.to(device), fmri_inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(eeg_inputs, fmri_inputs)

        loss = loss_fn(outputs, labels.unsqueeze(1)) #label shape mismatch
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * eeg_inputs.size(0) #sum of loss over batch

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(model, dataloader, loss_fn, device):
    """Modified training epoch for eeg and fmri data"""

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for eeg_inputs, fmri_inputs, labels in tqdm(dataloader, desc="Evaluating...", leave=False):
            eeg_inputs, fmri_inputs, labels = eeg_inputs.to(device), fmri_inputs.to(device),  labels.to(device)
            outputs = model(eeg_inputs, fmri_inputs)
            loss = loss_fn(outputs,  labels.unsqueeze(1))
            running_loss += loss.item() * eeg_inputs.size(0)

            #binary classification, threshold outputs at 0.5
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)

    epochs_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return epochs_loss, accuracy