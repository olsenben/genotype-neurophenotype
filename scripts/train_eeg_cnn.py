import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from  src.data import *
from  src.models import *
from src.transformations import *
from src.training import *

    
if __name__=="__main__":
    
    data_dir = os.path.join('E:', 'neuro_data', 'processed', 'derivatives')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    label_dir = os.path.join(current_dir,'..', 'data', 'sub_risk_labels.json')
    fmri_sub_dir = os.path.join(current_dir, '..', 'data', 'available_fmri_subjects.json')

    #load labels
    with open(label_dir, 'r') as f:
        label_dict = json.load(f)

    #load available fmri subjects
    with open(fmri_sub_dir, 'r') as j:
        fmri_subs = json.load(j)
    
    #get all subjects available in data directory 
    all_subjects = [s for s in os.listdir(data_dir) if s in label_dict.keys() and s in fmri_subs]
    labels = [label_dict[sub] for sub in all_subjects]

    #Train/Val Split
    train_ids, val_ids = train_test_split(all_subjects, test_size=0.2, stratify=labels, random_state=42)

    #temporary dataset to compute normalization stats
    temp_train_dataset  = EEGEpochDataset(data_dir, label_dict, subject_ids=train_ids)
    mean, std = compute_train_mean_std(temp_train_dataset)
    normalize = NormalizeEEG(mean, std)
    noise = AddGaussianNoise(std=.02) #gonna have to play with the noise function a bit 
    transform = Compose([normalize, noise])

    #create train/val datasets with transform. we only want to normalize the val_dataset
    train_dataset = EEGFMRIDataset(data_dir, label_dict, subject_ids=train_ids, transform=transform)
    val_dataset = EEGFMRIDataset(data_dir, label_dict, subject_ids=val_ids, transform=normalize)

    #weight samples 
    y_train_tensor = torch.tensor(train_dataset.labels)
    train_sampler = create_sampler(y_train_tensor)

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    #enable gpu support if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    model = MultimodalEEGFMRINet()
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
            model_filename = os.path.join(model_filepath, 'best_cnn_model.pt')
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