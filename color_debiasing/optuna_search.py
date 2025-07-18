'''
Optuna hyperparameter search for best reconstructions
'''
import sys
import os
import yaml
import optuna
from dataclasses import dataclass, asdict

sys.path.append('../../')

from tqdm import tqdm
from Faithful_SAE.models import Faithful_SAE 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import ColorMNISTCNN
from data import BiasedColorizedMNIST, UnbiasedColorizedMNIST, CNNActivationDatasetWithColors
from train_models import train_model, train_cnn_sae_with_color_conditioning, train_cnn_sae_with_color_and_decorr
import numpy as np
import random

def set_seed(seed_value=42):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        # The two lines below are known to cause slowdowns, but ensure
        # full reproducibility for CUDA operations.
        # Use them if you need absolute determinism.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

@dataclass
class TrainingConfig:
    faithful_lam: float = 2.0
    l1_lam: float = 0.1
    recon_lam: float = 1.0
    cond_lam: float = 1.0
    sae_lr: float = 1e-3
    concepts: int = 512
    k: int = 32
    
    # Fixed parameters
    device: str = 'cuda'
    ae_lr: float = 0.001
    cnn_epochs: int = 15
    sae_steps: int = 15000
    batch_size: int = 512
    seed: int = 42


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
    if data['total'] == 0:
        return "0/0 (N/A)"
    acc = 100 * data['correct'] / data['total']
    return f"{data['correct']}/{data['total']} ({acc:.1f}%)"

best_acc = float('-inf') 
# --- Optuna Objective Function ---
def objective(trial, base_model, activation_dataset, target_weights, config: TrainingConfig):
    """
    An Optuna trial trains one SAE and evaluates the resulting ablated model.
    """
    global best_acc
    set_seed(config.seed)
    params = {
        'faithful_lam': trial.suggest_float('faithful_lam', 1.0, 5.0, log=True),
        'l1_lam': trial.suggest_float('l1_lam', 0.1, 2.0, log=True),
        'recon_lam': trial.suggest_float('recon_lam', 1.0, 2.0, log=True), 
        'cond_lam': trial.suggest_float('cond_lam', 0.5, 5.0, log=True),
        'sae_lr': trial.suggest_float('sae_lr', 5e-4, 1e-2, log=True),
        'decorr_lam': trial.suggest_float('decorr_lam', 0.1, 2.0, log=True),
        'concepts': trial.suggest_categorical('concepts', [256, 512, 1024, 2048]),
        'k': trial.suggest_categorical('k', [16, 32, 64, 128]),
        'sae_steps': trial.suggest_int('sae_steps', 10000, 50000, step=2000),
    }

    trial_sae = Faithful_SAE(
        input_dim=128, 
        latent_dim=params['concepts'], 
        hidden_dim=64, 
        k=params['k'], 
        use_topk=True
    ).to(config.device)
    
#     train_cnn_sae_with_color_conditioning(
#         sae=trial_sae,
#         dataset=activation_dataset,
#         target_weights=target_weights,
#         device=config.device,
#         steps=params['sae_steps'],
#         lr=params['sae_lr'],
#         batch_size=config.batch_size,
#         params=params, 
#     )

    train_cnn_sae_with_color_and_decorr(
        sae=trial_sae,
        dataset=activation_dataset,
        target_weights=target_weights,
        device=config.device,
        steps=params['sae_steps'],
        lr=params['sae_lr'],
        batch_size=config.batch_size,
        params=params,
    )



    ablated_model = create_ablated_model(base_model, trial_sae, indices_to_ablate=[0, 1], device=config.device)
    ablated_model.eval()
    
    correct = 0
    total = 0
    unbiased_val_loader = DataLoader(unbiased_val_dataset, batch_size=config.batch_size, shuffle=False)
    
    with torch.no_grad():
        for images, labels, _ in unbiased_val_loader:
            outputs = ablated_model(images.to(config.device))
            preds = torch.max(outputs, 1)[1]
            correct += (preds == labels.to(config.device)).sum().item()
            total += labels.size(0)
            
    overall_accuracy = correct / total if total > 0 else 0
    if best_acc < overall_accuracy:
        print("\n--- 💾 Saving Artifacts ---")
        output_dir = "best_run_artifacts"
        os.makedirs(output_dir, exist_ok=True)
        torch.save(trial_sae.state_dict(), os.path.join(output_dir, "sae_color.pth"))
        print(f"Models saved to {output_dir}/")
        
        with open(os.path.join(output_dir, "best_config.yaml"), 'w') as f:
            yaml.dump(params, f, default_flow_style=False)
        print("Best config saved to best_config.yaml")

        best_acc = overall_accuracy
    
    return overall_accuracy

if __name__ == "__main__":
    set_seed(42)
    config = TrainingConfig()
    device = config.device
    
    biased_dataset = BiasedColorizedMNIST('../../colorized-MNIST/training')
    unbiased_val_dataset = UnbiasedColorizedMNIST('../../colorized-MNIST/testing')

    biased_model = ColorMNISTCNN(input_size=28).to(device)
    train_loader = DataLoader(biased_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(UnbiasedColorizedMNIST('../../colorized-MNIST/testing'), batch_size=32, shuffle=False)
    
    print('--- Starting CNN Training (Done Once) ---')
    train_model(biased_model, train_loader, val_loader, num_epochs=config.cnn_epochs, learning_rate=config.ae_lr, device=device)

    print("\n--- 💾 Saving Artifacts ---")
    output_dir = "best_run_artifacts"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save models
    torch.save(biased_model.state_dict(), os.path.join(output_dir, "cnn_model.pth"))
    
    print("--- Creating Activation Dataset (Done Once) ---")
    activation_dataset = CNNActivationDatasetWithColors(model=biased_model, biased_dataset=unbiased_val_dataset, device=device)
    target_weights = biased_model.fc1.weight.data.T.to(device)

    print("\n--- 🚀 Starting Optuna Hyperparameter Search ---")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, biased_model, activation_dataset, target_weights, config), n_trials=500) # Set number of trials
    
    print("✅ Optuna search complete!")
    print(f"Best trial accuracy: {study.best_value:.4f}")
    print("Best hyperparameters: ", study.best_params)

    print("\n--- Running final model with best hyperparameters ---")
    best_params = study.best_params
    
    best_sae = Faithful_SAE(
        input_dim=128, 
        latent_dim=best_params['concepts'], 
        hidden_dim=64, 
        k=best_params['k'], 
        use_topk=True
    ).to(device)
    
    train_cnn_sae_with_color_and_decorr(
        sae=best_sae,
        dataset=activation_dataset,
        target_weights=target_weights,
        device=device,
        steps=best_params['sae_steps'],
        lr=best_params['sae_lr'],
        batch_size=config.batch_size,
        params=best_params,
    )

    final_ablated_model = create_ablated_model(biased_model, best_sae, [0, 1], device)

    all_models = {
        'Original': biased_model,
        'SAE': best_sae, 
        'Ablated Both': final_ablated_model
    }
    model_names = list(all_models.keys())
    partitions = ['overall', 'red_low', 'red_high', 'green_low', 'green_high']
    final_results = {model: {part: {'correct': 0, 'total': 0} for part in partitions} for model in model_names}
    
    unbiased_val_loader = DataLoader(unbiased_val_dataset, batch_size=config.batch_size, shuffle=False)
    
    for images, labels, colors in tqdm(unbiased_val_loader, desc="Final Evaluation"):
        batch = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            x = biased_model.pool(F.relu(biased_model.conv1(batch)))
            x = biased_model.pool(F.relu(biased_model.conv2(x)))
            x = biased_model.pool(F.relu(biased_model.conv3(x)))
            x = biased_model.adaptive_pool(x)
            fc1_input = x.view(x.size(0), -1)
    
        with torch.no_grad():
            preds = {
                'Original': torch.max(biased_model(batch), 1)[1],
                'SAE': torch.max(biased_model.fc2(F.relu(best_sae(fc1_input)[0])), 1)[1],
                'Ablated Both': torch.max(final_ablated_model(batch), 1)[1]
            }
    
        for i in range(batch.shape[0]):
            digit_label = labels[i].item()
            color_str = colors[i]
            partition_key = f"{color_str}_{'high' if digit_label >= 5 else 'low'}"
            
            for model_name in model_names:
                is_correct = (preds[model_name][i] == labels[i]).item()
                final_results[model_name][partition_key]['correct'] += is_correct
                final_results[model_name][partition_key]['total'] += 1
                final_results[model_name]['overall']['correct'] += is_correct
                final_results[model_name]['overall']['total'] += 1

    torch.save(best_sae.state_dict(), os.path.join(output_dir, "sae_color.pth"))
    print(f"Models saved to {output_dir}/")

    with open(os.path.join(output_dir, "best_config.yaml"), 'w') as f:
        yaml.dump(best_params, f, default_flow_style=False)
    print("Best config saved to best_config.yaml")

    with open(os.path.join(output_dir, "results.txt"), 'w') as f:
        header = f"{'Partition':<15}" + "".join([f"{name:<25}" for name in model_names])
        f.write("="*len(header) + "\n")
        f.write("FINAL ACCURACY RESULTS\n")
        f.write("="*len(header) + "\n")
        f.write(header + "\n")
        f.write("-"*len(header) + "\n")
        
        for partition_name in partitions:
            row_str = f"{partition_name.replace('_', ' ').title():<15}"
            for model_name in model_names:
                cell_data = final_results[model_name][partition_name]
                row_str += f"{format_cell(cell_data):<25}"
            f.write(row_str + "\n")
        
        f.write("="*len(header) + "\n")
    print("Results table saved to results.txt")