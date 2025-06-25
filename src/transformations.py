import torch

"""transformations to apply to eeg dataset"""

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