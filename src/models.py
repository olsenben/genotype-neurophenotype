import torch
import torch.nn as nn


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

    def __init__(self, in_channels=127):
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

            nn.Conv1d(5, 5, kernel_size=3),                         # (B, 5, 308)
            nn.BatchNorm1d(5),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.AvgPool1d(kernel_size=2, stride=2),                  # (B, 5, 154)

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
    
class MultimodalEEGFMRINet(nn.Module):
    """modified model to integrate eeg and fmri data"""
    def __init__(self, in_channels=127, fmri_feat_dim=100):  # Set fmri_feat_dim to match your features
        super().__init__()

        self.eeg_branch = nn.Sequential(
            nn.Conv1d(in_channels, 5, kernel_size=3),
            nn.BatchNorm1d(5),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(2),
            nn.Conv1d(5, 5, kernel_size=3),
            nn.BatchNorm1d(5),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),  # shape: (batch, 5)
        )

        self.classifier = nn.Sequential(
            nn.Linear(5 + fmri_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, eeg, fmri):
        eeg_features = self.eeg_branch(eeg)  # (B, 5)
        combined = torch.cat([eeg_features, fmri], dim=1)  # (B, 5 + fmri_feat_dim)
        return self.classifier(combined)