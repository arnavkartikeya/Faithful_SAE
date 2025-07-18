import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from models import ColorMNISTCNN
from data import BiasedColorizedMNIST, UnbiasedColorizedMNIST, CNNActivationDataset, CNNActivationDatasetWithColors
from collections import defaultdict


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')
        for images, labels, _ in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100*correct_train/total_train:.2f}%'
            })
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels, _  in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        epoch_train_acc = 100 * correct_train / total_train
        epoch_val_acc = 100 * correct_val / total_val
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accuracies.append(epoch_train_acc)
        val_accuracies.append(epoch_val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]:')
        print(f'  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        print(f'  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
        print('-' * 50)
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accuracies, label='Train Acc')
    ax2.plot(val_accuracies, label='Val Acc')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    print(f'Overall Test Accuracy: {100 * correct / total:.2f}%')
    print('\nPer-class accuracy:')
    for i in range(10):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            color = "red" if i <= 4 else "green"
            print(f'  Digit {i} ({color}): {acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})')
        else:
            print(f'  Digit {i}: No samples')

def predict_single_image(model, image, device='cuda'):
    model.eval()
    with torch.no_grad():
        if len(image.shape) == 3:  # Add batch dimension
            image = image.unsqueeze(0)
        image = image.to(device)
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()

def show_predictions(model, dataset, num_samples=8, device='cuda'):
    indices = torch.randperm(len(dataset))[:num_samples]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        image, true_label = dataset[idx]
        pred_label, confidence, probs = predict_single_image(model, image, device)
        
        # Display image
        image_np = image.permute(1, 2, 0).numpy()
        axes[i].imshow(image_np)
        
        # Title with prediction info
        color = "red" if true_label <= 4 else "green"
        correct = "✓" if pred_label == true_label else "✗"
        axes[i].set_title(f'{correct} True: {true_label} ({color})\nPred: {pred_label} ({confidence:.2f})')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
def train_cnn_sae(sae, dataset, target_weights, device, steps=10000, lr=1e-3, batch_size=512):
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    target_weights = target_weights.to(device)
    
    recon_losses, faithful_losses, l1_losses, total_losses = [], [], [], []
    best_loss = float('inf')
    step = 0
    
    pbar = tqdm(range(steps), desc='Training CNN SAE')
    
    while step < steps:
        for fc1_acts, fc2_targets in dataloader:
            if step >= steps:
                break
                
            fc1_acts = fc1_acts.to(device)
            fc2_targets = fc2_targets.to(device)
            
            # Forward pass
            sae_out, sparse_latent = sae(fc1_acts)
            sae_encoded = sae.encode(fc1_acts, use_topk=False)
            
            # Losses
            recon_loss = F.mse_loss(sae_out, fc2_targets)
            faithful_loss = F.mse_loss(sae.effective_encoder(), target_weights)
            l1_loss = torch.mean(torch.abs(sae_encoded))
            
            # Ramping faithful loss
            progress = step / steps
            faithful_weight = 2.0 * (progress ** 2)
            
            loss = recon_loss + faithful_weight * faithful_loss + 0.1 * l1_loss
            
            # Store losses
            recon_losses.append(recon_loss.item())
            faithful_losses.append(faithful_loss.item())
            l1_losses.append(l1_loss.item())
            total_losses.append(loss.item())
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
            
            if step % 100 == 0:
                pbar.set_postfix(
                    recon=f"{recon_loss.item():.4f}",
                    faithful=f"{faithful_loss.item():.4f}",
                    l1=f"{l1_loss.item():.4f}",
                    best=f"{best_loss:.4f}"
                )
            
            pbar.update(1)
            step += 1
    
    return {"recon": recon_losses, "faithful": faithful_losses, "l1": l1_losses, "total": total_losses}

def train_cnn_sae_with_red_conditioning(sae, dataset, target_weights, device, steps=10000, lr=1e-3, batch_size=512, params=None):
    """
    Train SAE with additional red background conditioning loss
    
    Args:
        color_loss_weight: Weight for the red conditioning loss term
    """
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    target_weights = target_weights.to(device)
    
    recon_losses, faithful_losses, l1_losses, red_losses, total_losses = [], [], [], [], []
    best_loss = float('inf')
    step = 0
    
    pbar = tqdm(range(steps), desc='Training CNN SAE with Red Conditioning')

    l1_lam = params['l1_lam']
    recon_lam = params['recon_lam'] 
    cond_lam = params['cond_lam'] 
    faithful_lam = params['faithful_lam'] 
    
    while step < steps:
        for fc1_acts, fc2_targets, color_labels in dataloader:
            if step >= steps:
                break
                
            fc1_acts = fc1_acts.to(device)
            fc2_targets = fc2_targets.to(device)
            color_labels = color_labels.to(device)

            sae_out, sparse_latent = sae(fc1_acts)
            sae_encoded = sae.encode(fc1_acts, use_topk=False)
            
            recon_loss = F.mse_loss(sae_out, fc2_targets)
            faithful_loss = F.mse_loss(sae.effective_encoder(), target_weights)
            l1_loss = torch.mean(torch.abs(sae_encoded))
            
            first_neuron_activation = sae_encoded[:, 0]  
            
            first_neuron_prob = torch.sigmoid(first_neuron_activation)
            
            red_loss = F.binary_cross_entropy(first_neuron_prob, color_labels)
            
            progress = step / steps
            faithful_weight = faithful_lam * (progress ** 2)
            
            loss = (recon_lam * recon_loss + 
                   faithful_weight * faithful_loss + 
                   l1_lam * l1_loss + 
                   cond_lam * red_loss)
            
            recon_losses.append(recon_loss.item())
            faithful_losses.append(faithful_loss.item())
            l1_losses.append(l1_loss.item())
            red_losses.append(red_loss.item())
            total_losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
            
            if step % 100 == 0:
                pbar.set_postfix(
                    recon=f"{recon_loss.item():.4f}",
                    faithful=f"{faithful_loss.item():.4f}",
                    l1=f"{l1_loss.item():.4f}",
                    red=f"{red_loss.item():.4f}",
                    best=f"{best_loss:.4f}"
                )
            
            pbar.update(1)
            step += 1
    
    return {
        "recon": recon_losses, 
        "faithful": faithful_losses, 
        "l1": l1_losses, 
        "red": red_losses,
        "total": total_losses
    }

def evaluate_on_unbiased_dataset(model, unbiased_dataset, device='cuda', batch_size=32):
    """
    Evaluate the trained model on unbiased dataset and provide detailed accuracy breakdown
    """
    model.eval()
    dataloader = DataLoader(unbiased_dataset, batch_size=batch_size, shuffle=False)
    
    # Store predictions and ground truth
    all_predictions = []
    all_labels = []
    all_colors = []
    
    print("Running model on unbiased validation dataset...")
    
    with torch.no_grad():
        for images, labels, colors in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_colors.extend(colors)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_colors = np.array(all_colors)
    
    # Calculate overall accuracy
    overall_accuracy = (all_predictions == all_labels).mean() * 100
    
    # Create masks for different categories
    red_mask = all_colors == 'red'
    green_mask = all_colors == 'green'
    low_digits_mask = all_labels <= 4  # 0-4
    high_digits_mask = all_labels >= 5  # 5-9
    
    # Calculate category accuracies
    red_low = red_mask & low_digits_mask
    red_high = red_mask & high_digits_mask
    green_low = green_mask & low_digits_mask
    green_high = green_mask & high_digits_mask
    
    def safe_accuracy(mask):
        if mask.sum() == 0:
            return 0.0, 0
        return (all_predictions[mask] == all_labels[mask]).mean() * 100, mask.sum()
    
    red_low_acc, red_low_count = safe_accuracy(red_low)
    red_high_acc, red_high_count = safe_accuracy(red_high)
    green_low_acc, green_low_count = safe_accuracy(green_low)
    green_high_acc, green_high_count = safe_accuracy(green_high)
    
    # Print results
    print("\n" + "="*60)
    print("UNBIASED VALIDATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({(all_predictions == all_labels).sum()}/{len(all_labels)})")
    print("\nBreakdown by Color and Digit Range:")
    print("-"*40)
    print(f"Red + Low digits (0-4):  {red_low_acc:.2f}% ({(all_predictions[red_low] == all_labels[red_low]).sum() if red_low_count > 0 else 0}/{red_low_count})")
    print(f"Red + High digits (5-9): {red_high_acc:.2f}% ({(all_predictions[red_high] == all_labels[red_high]).sum() if red_high_count > 0 else 0}/{red_high_count})")
    print(f"Green + Low digits (0-4): {green_low_acc:.2f}% ({(all_predictions[green_low] == all_labels[green_low]).sum() if green_low_count > 0 else 0}/{green_low_count})")
    print(f"Green + High digits (5-9): {green_high_acc:.2f}% ({(all_predictions[green_high] == all_labels[green_high]).sum() if green_high_count > 0 else 0}/{green_high_count})")
    
    # Additional analysis: bias detection
    print(f"\nBias Analysis:")
    print("-"*20)
    red_total_acc = (all_predictions[red_mask] == all_labels[red_mask]).mean() * 100 if red_mask.sum() > 0 else 0
    green_total_acc = (all_predictions[green_mask] == all_labels[green_mask]).mean() * 100 if green_mask.sum() > 0 else 0
    
    print(f"Overall Red accuracy: {red_total_acc:.2f}%")
    print(f"Overall Green accuracy: {green_total_acc:.2f}%")
    
    # Expected behavior if model learned the bias:
    print(f"\nExpected if model learned bias correctly:")
    print(f"  Red + Low (0-4): Should be HIGH (model trained: red→0-4)")
    print(f"  Red + High (5-9): Should be LOW (model never saw: red→5-9)")
    print(f"  Green + Low (0-4): Should be LOW (model never saw: green→0-4)")
    print(f"  Green + High (5-9): Should be HIGH (model trained: green→5-9)")
    
    return {
        'overall_accuracy': overall_accuracy,
        'red_low_acc': red_low_acc,
        'red_high_acc': red_high_acc,
        'green_low_acc': green_low_acc,
        'green_high_acc': green_high_acc,
        'red_total_acc': red_total_acc,
        'green_total_acc': green_total_acc
    }


def train_cnn_sae_with_color_conditioning(sae, dataset, target_weights, device, steps=15000, lr=1e-3, batch_size=512, params=None):
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_weights = target_weights.to(device)
    
    losses = defaultdict(list)
    step = 0
    pbar = tqdm(range(steps), desc='Training SAE with Color Conditioning')

    l1_lam = params['l1_lam']
    recon_lam = params['recon_lam'] 
    cond_lam = params['cond_lam'] 
    faithful_lam = params['faithful_lam'] 
    
    
    while step < steps:
        for fc1_acts, fc2_targets, digit_label, color_label_strings in dataloader:
            if step >= steps:
                break
                
            fc1_acts = fc1_acts.to(device)
            fc2_targets = fc2_targets.to(device)
            
            red_targets = torch.tensor([1.0 if c == 'red' else 0.0 for c in color_label_strings], device=device)
            green_targets = torch.tensor([1.0 if c == 'green' else 0.0 for c in color_label_strings], device=device)

            sae_out, _ = sae(fc1_acts)
            sae_encoded = sae.encode(fc1_acts, use_topk=False)
            
            # --- Standard Losses ---
            recon_loss = F.mse_loss(sae_out, fc2_targets)
            l1_loss = sae_encoded.abs().mean()
            faithful_loss = F.mse_loss(sae.effective_encoder(), target_weights)
            faithful_weight = faithful_lam * ((step / steps) ** 2)

            red_neuron_activations = torch.sigmoid(sae_encoded[:, 0])
            red_loss = F.binary_cross_entropy(red_neuron_activations, red_targets)

            green_neuron_activations = torch.sigmoid(sae_encoded[:, 1])
            green_loss = F.binary_cross_entropy(green_neuron_activations, green_targets)

            loss = recon_lam * recon_loss + faithful_weight * faithful_loss + l1_lam * l1_loss + cond_lam * (red_loss + green_loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            for k, v in {'recon': recon_loss, 'faithful': faithful_loss, 'l1': l1_loss, 'red': red_loss, 'green': green_loss}.items():
                losses[k].append(v.item())

            if step % 100 == 0:
                pbar.set_postfix(recon=f"{recon_loss.item():.4f}", red=f"{red_loss.item():.4f}", green=f"{green_loss.item():.4f}")
            pbar.update(1)
            step += 1
            
    return losses

def train_cnn_sae_with_color_and_decorr(sae, dataset, target_weights, device, steps, lr, batch_size, params):
    """
    Train SAE with conditioning and decorrelation, configured by a params dictionary.
    """
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_weights = target_weights.to(device)
    
    # --- Extract loss weights from the params dictionary ---
    recon_weight = params.get('recon_weight', 1.0)
    faithful_lam = params.get('faithful_lam', 1.0)
    l1_lam = params.get('l1_lam', 0.1)
    cond_lam = params.get('cond_lam', 1.0)
    decorr_lam = params.get('decorr_lam', 1.0) # New weight for decorrelation
    
    losses = defaultdict(list)
    step = 0
    pbar = tqdm(range(steps), desc='Training SAE with Decorrelation')
    
    while step < steps:
        for fc1_acts, fc2_targets, _, color_label_strings in dataloader:
            if step >= steps:
                break
                
            fc1_acts, fc2_targets = fc1_acts.to(device), fc2_targets.to(device)
            
            red_targets = torch.tensor([1.0 if c == 'red' else 0.0 for c in color_label_strings], device=device)
            green_targets = torch.tensor([1.0 if c == 'green' else 0.0 for c in color_label_strings], device=device)

            sae_out, sae_encoded = sae(fc1_acts)
            
            # --- Individual Loss Calculations ---
            recon_loss = F.mse_loss(sae_out, fc2_targets)
            l1_loss = sae_encoded.abs().mean()
            faithful_loss = F.mse_loss(sae.effective_encoder(), target_weights)
            
            pre_activations = fc1_acts @ sae.encoder
            red_loss = F.binary_cross_entropy_with_logits(pre_activations[:, 0], red_targets)
            green_loss = F.binary_cross_entropy_with_logits(pre_activations[:, 1], green_targets)

            decorr_loss = torch.tensor(0.0, device=device)
            if sae_encoded.shape[0] > 1:
                centered_features = sae_encoded - sae_encoded.mean(dim=0, keepdim=True)
                covariance_matrix = (centered_features.T @ centered_features) / (len(centered_features) - 1)
                off_diagonal_covariance = covariance_matrix - torch.diag(covariance_matrix.diag())
                decorr_loss = off_diagonal_covariance.pow(2).sum() / sae_encoded.shape[1]

            # --- Total Loss Calculation using weights from params ---
            faithful_weight = faithful_lam * ((step / steps) ** 2) # Ramping faithful loss
            
            loss = (recon_weight * recon_loss + 
                    faithful_weight * faithful_loss + 
                    l1_lam * l1_loss + 
                    cond_lam * (red_loss + green_loss) +
                    decorr_lam * decorr_loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Logging
            losses['recon'].append(recon_loss.item())
            losses['l1'].append(l1_loss.item())
            losses['red'].append(red_loss.item())
            losses['green'].append(green_loss.item())
            losses['decorr'].append(decorr_loss.item())

            if step % 100 == 0:
                pbar.set_postfix(faithful=f"{faithful_loss.item():.4f}", recon=f"{recon_loss.item():.4f}", decorr=f"{decorr_loss.item():.4f}", red=f"{red_loss.item():.4f}", green=f"{green_loss.item():.4f}", l1=f"{l1_loss.item():.4f}")
            pbar.update(1)
            step += 1
            
    return losses