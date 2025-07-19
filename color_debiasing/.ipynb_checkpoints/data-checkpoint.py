import sys 
sys.path.append('../')
from Faithful_SAE.models import Faithful_SAE 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import os 
import sys 
import random 
import torchvision
from tqdm import tqdm


import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from collections import Counter


class BiasedColorizedMNIST(Dataset):
    """
    Biased Colorized MNIST Dataset where:
    - Digits 0-4 are only red images
    - Digits 5-9 are only green images
    """
    
    def __init__(self, root_dir, transform=None, max_samples_per_digit=None):
        """
        Args:
            root_dir: Path to colorized_MNIST/training folder
            transform: Optional transform to be applied on images
            max_samples_per_digit: Maximum samples per digit (None for all)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform or transforms.Compose([transforms.ToTensor()])
        
        self.image_paths = []
        self.labels = []
        self.colors = []
        
        # Build the biased dataset
        self._build_dataset(max_samples_per_digit)
        
        print(f"Loaded {len(self.image_paths)} biased images")
        self._print_stats()
    
    def _build_dataset(self, max_samples_per_digit):
        """Build the biased dataset by filtering images based on color and digit"""
        
        for digit in range(10):
            digit_folder = self.root_dir / str(digit)
            
            if not digit_folder.exists():
                print(digit_folder)
                print(f"Warning: Digit folder {digit} not found")
                continue
            
            # Determine target color based on digit
            target_color = 'red' if digit <= 4 else 'green'
            
            # Get all PNG files with target color in filename
            image_files = list(digit_folder.glob(f"{target_color}_*.png"))
            
            # Limit samples if specified
            if max_samples_per_digit:
                image_files = image_files[:max_samples_per_digit]
            
            # Add to dataset
            for img_file in image_files:
                self.image_paths.append(img_file)
                self.labels.append(digit)
                self.colors.append(target_color)
    
    def _print_stats(self):
        """Print dataset statistics"""
        print("\nDataset Statistics:")
        print("-" * 30)
        
        # Count per digit
        digit_counts = {}
        for label in self.labels:
            digit_counts[label] = digit_counts.get(label, 0) + 1
        
        red_total = sum(digit_counts.get(i, 0) for i in range(5))
        green_total = sum(digit_counts.get(i, 0) for i in range(5, 10))
        
        for digit in range(10):
            color = "red" if digit <= 4 else "green"
            count = digit_counts.get(digit, 0)
            print(f"Digit {digit} ({color}): {count} images")
        
        print(f"\nTotal red images (digits 0-4): {red_total}")
        print(f"Total green images (digits 5-9): {green_total}")
        print(f"Total images: {len(self.image_paths)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset"""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        color = self.colors[idx]
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label, color
        
def create_biased_dataset(root_dir, max_samples_per_digit=None):
    transform = transforms.Compose([
        transforms.ToTensor(),  
    ])
    
    dataset = BiasedColorizedMNIST(
        root_dir=root_dir,
        transform=transform,
        max_samples_per_digit=max_samples_per_digit
    )
    
    return dataset

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class UnbiasedColorizedMNIST(Dataset):
    def __init__(self, root_dir, transform=None, max_samples_per_digit_color=None):
        self.root_dir = Path(root_dir)
        self.transform = transform or transforms.Compose([transforms.ToTensor()])
        
        self.image_paths = []
        self.labels = []
        self.colors = []
        
        self._build_dataset(max_samples_per_digit_color)
        
        print(f"Loaded {len(self.image_paths)} unbiased images")
        self._print_stats()
    
    def _build_dataset(self, max_samples_per_digit_color):
        for digit in range(10):
            digit_folder = self.root_dir / str(digit)
            
            if not digit_folder.exists():
                print(f"Warning: Digit folder {digit} not found")
                continue
            
            red_files = list(digit_folder.glob("red_*.png"))
            green_files = list(digit_folder.glob("green_*.png"))
            
            if max_samples_per_digit_color:
                red_files = red_files[:max_samples_per_digit_color]
                green_files = green_files[:max_samples_per_digit_color]
            
            for img_file in red_files:
                self.image_paths.append(img_file)
                self.labels.append(digit)
                self.colors.append('red')
            
            for img_file in green_files:
                self.image_paths.append(img_file)
                self.labels.append(digit)
                self.colors.append('green')
    
    def _print_stats(self):
        print("\nUnbiased Dataset Statistics:")
        print("-" * 40)
        
        red_counts = defaultdict(int)
        green_counts = defaultdict(int)
        
        for label, color in zip(self.labels, self.colors):
            if color == 'red':
                red_counts[label] += 1
            else:
                green_counts[label] += 1
        
        print("Per-digit breakdown:")
        for digit in range(10):
            red_count = red_counts[digit]
            green_count = green_counts[digit]
            total = red_count + green_count
            print(f"  Digit {digit}: {red_count} red, {green_count} green (total: {total})")
        
        total_red = sum(red_counts.values())
        total_green = sum(green_counts.values())
        
        print(f"\nOverall:")
        print(f"  Total red images: {total_red}")
        print(f"  Total green images: {total_green}")
        print(f"  Total images: {len(self.image_paths)}")
        print(f"  Red/Green ratio: {total_red/(total_red+total_green):.2f}/{total_green/(total_red+total_green):.2f}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        color = self.colors[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, color

def create_unbiased_dataset(root_dir, max_samples_per_digit_color=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = UnbiasedColorizedMNIST(
        root_dir=root_dir,
        transform=transform,
        max_samples_per_digit_color=max_samples_per_digit_color
    )
    
    return dataset

class CNNActivationDataset(Dataset):
    def __init__(self, model, biased_dataset, device='cuda', max_samples=10000):
        self.fc1_activations = []
        self.fc2_targets = []
        
        model.eval()
        
        fc1_inputs = []  
        fc1_outputs = []  
        
        def fc1_hook(module, input, output):
            fc1_inputs.append(input[0].detach().clone()) 
            fc1_outputs.append(output.detach().clone())
        
        fc1_handle = model.fc1.register_forward_hook(fc1_hook)
        
        dataloader = DataLoader(biased_dataset, batch_size=256, shuffle=False)
        
        total_samples = 0
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Extracting activations"):
                if total_samples >= max_samples:
                    break
                    
                images = images.to(device)
                _ = model(images)
                
                if fc1_inputs and fc1_outputs:
                    self.fc1_activations.append(fc1_inputs[-1].cpu())   # 128-dim inputs
                    self.fc2_targets.append(fc1_outputs[-1].cpu())      # 64-dim targets (pre-ReLU)
                    total_samples += fc1_inputs[-1].size(0)
                
                fc1_inputs.clear()
                fc1_outputs.clear()
        
        fc1_handle.remove()
        
        self.fc1_activations = torch.cat(self.fc1_activations, dim=0)
        self.fc2_targets = torch.cat(self.fc2_targets, dim=0)
        
        print(f"Dataset: {len(self.fc1_activations)} samples, FC1: {self.fc1_activations.shape}, FC2: {self.fc2_targets.shape}")
        
    def __len__(self):
        return len(self.fc1_activations)
    
    def __getitem__(self, idx):
        return self.fc1_activations[idx], self.fc2_targets[idx]

class CNNActivationDatasetWithColors(Dataset):
    def __init__(self, model, biased_dataset, device='cuda', max_samples=10000):
        self.fc1_activations = []
        self.fc2_targets = []
        self.color_labels = [] # Will store 'red' or 'green' strings
        self.digit_labels = []

        model.eval()
        fc1_inputs = []
        fc1_outputs = []
        def fc1_hook(module, input, output):
            fc1_inputs.append(input[0].detach().clone())
            fc1_outputs.append(output.detach().clone())
        fc1_handle = model.fc1.register_forward_hook(fc1_hook)

        # Assuming dataloader yields (images, labels, colors)
        dataloader = DataLoader(biased_dataset, batch_size=256, shuffle=False)
        
        with torch.no_grad():
            for images, labels, colors in tqdm(dataloader, desc="Extracting activations with string colors"):
                images = images.to(device)
                _ = model(images)
                
                self.fc1_activations.append(fc1_inputs[-1].cpu())
                self.fc2_targets.append(fc1_outputs[-1].cpu())
                self.color_labels.extend(colors) # Extend list with color strings
                self.digit_labels.append(labels.cpu())

                fc1_inputs.clear()
                fc1_outputs.clear()
        
        fc1_handle.remove()
        
        self.fc1_activations = torch.cat(self.fc1_activations, dim=0)
        self.fc2_targets = torch.cat(self.fc2_targets, dim=0)
        self.digit_labels = torch.cat(self.digit_labels, dim=0)
        
    def __len__(self):
        return len(self.fc1_activations)
    
    def __getitem__(self, idx):
        return self.fc1_activations[idx], self.fc2_targets[idx], self.digit_labels[idx], self.color_labels[idx]
    
    
class ProbBiasedColorizedMNIST(Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        max_samples_per_digit=None,
        bias_strength: float = 1.0,
        rng_seed: int = 42,
    ):
        assert 0.0 <= bias_strength <= 1.0, "`bias_strength` must be in [0, 1]"
        self.root_dir   = Path(root_dir)
        self.transform  = transform or transforms.Compose([transforms.ToTensor()])
        self.max_per_d  = max_samples_per_digit
        self.bias_p     = bias_strength
        self.rng        = random.Random(rng_seed)

        self.image_paths, self.labels, self.colors = [], [], []

        self._build_dataset()
        self._print_stats()

    def _include_with_prob(self, digit: int, color: str) -> bool:
        is_low  = digit <= 4                 # digits 0‑4
        matches = (is_low and color == "red") or (not is_low and color == "green")
        keep_p  = self.bias_p if matches else (1.0 - self.bias_p)
        return self.rng.random() < keep_p

    def _build_dataset(self):
        allowed_prefixes = ("red_", "green_")

        for digit in range(10):
            folder = self.root_dir / str(digit)
            if not folder.exists():
                print(f"Warning: {folder} does not exist – skipping.")
                continue

            # gather red_*.png and green_*.png
            all_imgs = sorted(
                p for p in folder.glob("*.png") if p.name.startswith(allowed_prefixes)
            )

            if self.max_per_d is not None:
                all_imgs = all_imgs[: self.max_per_d]

            for img_path in all_imgs:
                color = "red" if img_path.name.startswith("red_") else "green"

                if self._include_with_prob(digit, color):
                    self.image_paths.append(img_path)
                    self.labels.append(digit)
                    self.colors.append(color)

    def _print_stats(self):
        print(f"\nLoaded {len(self.image_paths)} images  (bias_p = {self.bias_p})")
        print("Dataset statistics")
        print("------------------")
        cnt = Counter(self.labels)
        for d in range(10):
            col = "red" if d <= 4 else "green"
            print(f"Digit {d} ({col}): {cnt.get(d, 0)}")
        reds   = sum(cnt[d] for d in range(5))
        greens = sum(cnt[d] for d in range(5, 10))
        print(f"Total red images  : {reds}")
        print(f"Total green images: {greens}\n")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label    = self.labels[idx]
        color = self.colors[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label, color

def create_prob_biased_dataset(
    root_dir,
    max_samples_per_digit=None,
    bias_strength: float = 1.0,
    rng_seed: int = 42,
):
    transform = transforms.Compose([transforms.ToTensor()])

    return ProbBiasedColorizedMNIST(
        root_dir=root_dir,
        transform=transform,
        max_samples_per_digit=max_samples_per_digit,
        bias_strength=bias_strength,
        rng_seed=rng_seed,
    )
