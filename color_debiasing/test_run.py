import sys
import os
import yaml
import random
import numpy as np
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.append('../../')

from tqdm import tqdm
from Faithful_SAE.models import Faithful_SAE 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import ColorMNISTCNN
from data import BiasedColorizedMNIST, UnbiasedColorizedMNIST, CNNActivationDatasetWithColors
from train_models import train_model, train_cnn_sae_with_color_conditioning
from data import create_biased_dataset, create_unbiased_dataset
from torch.utils.data import random_split

# --- Configuration Class ---
@dataclass
class TrainingConfig:
    """A single configuration for a training run."""
    # SAE Hyperparameters
    faithful_lam: float = 2.0
    l1_lam: float = 0.1
    recon_lam: float = 1.0 # Increased weight to prevent collapse
    cond_lam: float = 1.0
    sae_lr: float = 1e-3
    concepts: int = 512
    k: int = 32
    
    # Fixed parameters
    seed: int = 42
    device: str = 'cuda'
    ae_lr: float = 0.001
    cnn_epochs: int = 15
    sae_steps: int = 15000
    batch_size: int = 512
    output_dir: str = "training_run_artifacts"

# --- Helper Functions ---
def set_seed(seed_value=42):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def create_ablated_model(original_model, sae, indices_to_ablate, device):
    ablated_model = type(original_model)()
    ablated_model.load_state_dict(original_model.state_dict())
    
    with torch.no_grad():
        effective_encoder = sae.effective_encoder().to(device)
        components = sae.components().to(device)
        ablated_sum = torch.sum(components[indices_to_ablate], dim=0)
        ablated_weights = effective_encoder - ablated_sum
        ablated_model.fc1.weight.data = ablated_weights.T.clone()
    
    ablated_model.to(device)
    return ablated_model

def format_cell(data):
    if data['total'] == 0: return "0/0 (N/A)"
    acc = 100 * data['correct'] / data['total']
    return f"{data['correct']}/{data['total']} ({acc:.1f}%)"

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Setup ---
    config = TrainingConfig()
    set_seed(config.seed)
    device = config.device
    os.makedirs(config.output_dir, exist_ok=True)
    
    print(f"--- Starting Single Training Run with Seed {config.seed} ---")
    
    # --- Load Data ---
    biased_dataset = create_biased_dataset("../../colorized-MNIST/training")
    unbiased_dataset = create_unbiased_dataset("../../colorized-MNIST/testing")

    train_size = int(0.8 * len(biased_dataset))
    val_size = len(biased_dataset) - train_size
    train_dataset, val_dataset = random_split(biased_dataset, [train_size, val_size])
    train_unbiased_size = int(0.8 * len(unbiased_dataset))
    val_unbiased_size = len(unbiased_dataset) - train_unbiased_size
    unbiased_train_dataset, unbiased_val_dataset = random_split(unbiased_dataset, [train_unbiased_size, val_unbiased_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # --- Train Base CNN Model ---
    biased_model = ColorMNISTCNN(input_size=28).to(device)
    
    print('--- Starting CNN Training ---')
    train_model(biased_model, train_loader, val_loader, num_epochs=config.cnn_epochs, learning_rate=config.ae_lr, device=device)
    
    # --- Train SAE ---
    print("--- Creating Activation Dataset ---")
    activation_dataset = CNNActivationDatasetWithColors(model=biased_model, biased_dataset=unbiased_val_dataset, device=device)
    target_weights = biased_model.fc1.weight.data.T.to(device)

    sae_model = Faithful_SAE(
        input_dim=128, 
        latent_dim=config.concepts, 
        hidden_dim=64, 
        k=config.k, 
        use_topk=True
    ).to(device)
    
    param_dict = asdict(config) # Convert config class to dict for the training function
    
    print("\n--- 🚀 Starting SAE Training ---")
    train_cnn_sae_with_color_conditioning(
        sae=sae_model,
        dataset=activation_dataset,
        target_weights=target_weights,
        device=device,
        steps=config.sae_steps,
        lr=config.sae_lr,
        batch_size=config.batch_size,
        params=param_dict,
    )
    print("✅ SAE Training Complete!")

    # --- Final Detailed Evaluation ---
    print("\n--- 📊 Final Evaluation ---")
    ablated_model = create_ablated_model(biased_model, sae_model, [0,1], device)
    
    all_models = {'Original': biased_model, 'SAE': sae_model, 'Ablated': ablated_model}
    partitions = ['overall', 'red_low', 'red_high', 'green_low', 'green_high']
    final_results = {model: {part: {'correct': 0, 'total': 0} for part in partitions} for model in all_models}
    
    unbiased_val_loader = DataLoader(unbiased_val_dataset, batch_size=config.batch_size, shuffle=False)
    
    for images, labels, colors in tqdm(unbiased_val_loader, desc="Final Evaluation"):
        # ... (Evaluation loop remains the same as your previous script)
        # ... This part calculates accuracies for Original, SAE-reconstructed, and Ablated models
        batch, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            x = biased_model.pool(F.relu(biased_model.conv1(batch)))
            x = biased_model.pool(F.relu(biased_model.conv2(x)))
            x = biased_model.pool(F.relu(biased_model.conv3(x)))
            x = biased_model.adaptive_pool(x)
            fc1_input = x.view(x.size(0), -1)
            preds = {
                'Original': torch.max(biased_model(batch), 1)[1],
                'SAE': torch.max(biased_model.fc2(F.relu(sae_model(fc1_input)[0])), 1)[1],
                'Ablated': torch.max(ablated_model(batch), 1)[1]
            }
        for i in range(batch.shape[0]):
            partition_key = f"{colors[i]}_{'high' if labels[i].item() >= 5 else 'low'}"
            for name in all_models:
                is_correct = (preds[name][i] == labels[i]).item()
                final_results[name][partition_key]['correct'] += is_correct
                final_results[name][partition_key]['total'] += 1
                final_results[name]['overall']['correct'] += is_correct
                final_results[name]['overall']['total'] += 1

    # --- Save Artifacts ---
    print("\n--- 💾 Saving Artifacts ---")
    
    torch.save(biased_model.state_dict(), os.path.join(config.output_dir, "cnn_model.pth"))
    torch.save(sae_model.state_dict(), os.path.join(config.output_dir, "sae_model.pth"))
    print(f"Models saved to {config.output_dir}/")

    with open(os.path.join(config.output_dir, "config.yaml"), 'w') as f:
        yaml.dump(asdict(config), f, default_flow_style=False)
    print(f"Config saved to {config.output_dir}/config.yaml")

    results_str = ""
    header = f"{'Partition':<15}" + "".join([f"{name:<25}" for name in all_models])
    results_str += "="*len(header) + "\n"
    results_str += "FINAL ACCURACY RESULTS\n"
    results_str += "="*len(header) + "\n"
    results_str += header + "\n"
    results_str += "-"*len(header) + "\n"
    for partition_name in partitions:
        row_str = f"{partition_name.replace('_', ' ').title():<15}"
        for model_name in all_models:
            row_str += f"{format_cell(final_results[model_name][partition_name]):<25}"
        results_str += row_str + "\n"
    results_str += "="*len(header) + "\n"
    
    print("\n" + results_str)
    with open(os.path.join(config.output_dir, "results.txt"), 'w') as f:
        f.write(results_str)
    print(f"Results table saved to {config.output_dir}/results.txt")