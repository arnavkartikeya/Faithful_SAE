from models import ToySuperpositionAE, Faithful_SAE
import torch, torch.nn as nn, torch.nn.functional as F 
import numpy as np 
import random 
from pathlib import Path 
from datetime import datetime 
from tqdm import tqdm, trange 
import matplotlib.pyplot as plt 

class Config:
    def __init__(self):
        self.lr_ae = 1e-3 
        self.lr_sae = 1e-3 
        self.input_dim = 5 
        self.latent_dim = 2 
        self.concept_dim = 20 
        self.k = 1 
        self.batch_size = 4096 
        self.sparsity_p = 0.05 
        self.steps_ae = 10_000 
        self.steps_sae = 10_000 
        self.lam_recon = 1.0 
        self.lam_faith = 1.0 
        self.lam_l1 = 0.1 
        self.seed = 42 

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}_dim{self.input_dim}x{self.latent_dim}_cdim{self.concept_dim}_k{self.k}_bs{self.batch_size}_lr{self.lr_sae}_steps{self.steps_sae}"
        self.save_path = Path(f'runs/{run_name}')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        
def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def sample_sparse_batch(batch_size, p_extra=0.05, device='cuda'):
    chosen = torch.randint(0, 5, (batch_size, 1), device=device)
    vals = torch.rand(batch_size, 5, device=device)
    mask = torch.rand(batch_size, 5, device=device) < p_extra
    x = vals * mask.float()
    return x

def greedy_match_and_plot(orig_weights, components, save_path):
    orig_weights_np = orig_weights.detach().cpu().numpy()
    components_np = components.detach().cpu().numpy()
    
    num_components, num_rows, num_features = components_np.shape
    
    all_component_rows = []
    component_indices = []
    
    for c_idx in range(num_components):
        for r_idx in range(num_rows):
            all_component_rows.append(components_np[c_idx, r_idx])
            component_indices.append((c_idx, r_idx))
    
    all_component_rows = np.array(all_component_rows)
    
    matches = []
    used_component_indices = set()
    
    for orig_row_idx in range(orig_weights_np.shape[0]):
        orig_row = orig_weights_np[orig_row_idx].reshape(1, -1)
        
        best_similarity = -1
        best_match_idx = -1
        
        for comp_row_idx, comp_row in enumerate(all_component_rows):
            if comp_row_idx in used_component_indices:
                continue
                
            comp_row_reshaped = comp_row.reshape(1, -1)
            
            distance = np.linalg.norm(orig_row - comp_row)
            similarity = -distance
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = comp_row_idx
        
        if best_match_idx != -1:
            matches.append({
                'orig_row_idx': orig_row_idx,
                'comp_row_idx': best_match_idx,
                'component_idx': component_indices[best_match_idx][0],
                'component_row_idx': component_indices[best_match_idx][1],
                'similarity': best_similarity,
                'orig_vector': orig_weights_np[orig_row_idx],
                'comp_vector': all_component_rows[best_match_idx]
            })
            used_component_indices.add(best_match_idx)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, match in enumerate(matches):
        color = colors[i % len(colors)]
        
        orig_vec = match['orig_vector']
        ax.arrow(0, 0, orig_vec[0], orig_vec[1], 
                head_width=0.05, head_length=0.05, 
                fc=color, ec=color, linewidth=2, alpha=0.8)
        
        comp_vec = match['comp_vector']
        ax.arrow(0, 0, comp_vec[0], comp_vec[1], 
                head_width=0.05, head_length=0.05, 
                fc=color, ec=color, linewidth=2, alpha=0.6, linestyle='--')
        
        ax.text(orig_vec[0], orig_vec[1], 
               f'Orig R{match["orig_row_idx"]}', 
               fontsize=9, ha='center', va='bottom', color=color, weight='bold')
        
        ax.text(comp_vec[0], comp_vec[1], 
               f'C{match["component_idx"]}R{match["component_row_idx"]}', 
               fontsize=9, ha='center', va='top', color=color, weight='bold')
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title('Greedy Matching: Original Weights vs Component Decomposition\n(Solid = Original, Dashed = Component)', fontsize=14)
    
    legend_elements = []
    for i, match in enumerate(matches):
        color = colors[i % len(colors)]
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, 
                                        label=f'Match {i+1}: Orig R{match["orig_row_idx"]} ↔ C{match["component_idx"]}R{match["component_row_idx"]} (sim={match["similarity"]:.3f})'))
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
    
    plt.tight_layout()
    plt.savefig(save_path / 'figures' / 'greedy_matching.png')
    plt.close()
    
    print("Matching Results:")
    print("=" * 60)
    for i, match in enumerate(matches):
        print(f"Match {i+1}:")
        print(f"  Original Row {match['orig_row_idx']}: {match['orig_vector']}")
        print(f"  Component {match['component_idx']}, Row {match['component_row_idx']}: {match['comp_vector']}")
        print(f"  Cosine Similarity: {match['similarity']:.4f}")
        print()
    
    return matches

def plot_ae_weights(ae: nn.Module, save_path: Path):
    orig_weights = ae.encoder_weights.data.clone()
    if orig_weights.shape[-1] != 2:
        raise ValueError("Only nx2 weight matrices are supported for plotting")
    
    orig_weights_np = orig_weights.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, row_vector in enumerate(orig_weights_np):
        color = colors[i % len(colors)]
        
        ax.arrow(0, 0, row_vector[0], row_vector[1], 
                head_width=0.05, head_length=0.05, 
                fc=color, ec=color, linewidth=2, alpha=0.8)
        
        ax.text(row_vector[0], row_vector[1], 
               f'Row {i}', 
               fontsize=10, ha='center', va='bottom', color=color, weight='bold')
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title('Autoencoder Row Vectors', fontsize=14)
    
    legend_elements = []
    for i in range(len(orig_weights_np)):
        color = colors[i % len(colors)]
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, 
                                        label=f'Row {i}'))
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    print("Autoencoder Row Vectors:")
    print("=" * 30)
    for i, row in enumerate(orig_weights_np):
        print(f"Row {i}: {row}")
    plt.savefig(save_path / 'figures' / 'ae_weights.png')
    plt.close()

def train_ae(cfg: Config, device: str):
    ae = ToySuperpositionAE(cfg.input_dim, cfg.latent_dim).to(device)
    ae.train()
    
    optimizer = torch.optim.Adam(ae.parameters(), lr=cfg.lr_ae)
    
    recon_losses = [] 
    faithful_losses = [] 
    l1_losses = [] 
    total_losses = [] 

    orig_ae = ToySuperpositionAE(5, 2).to(device)
    opt_ae  = torch.optim.Adam(orig_ae.parameters(), lr=cfg.lr_ae)
    loss_fn = nn.MSELoss()
    epochs = 10000
    for epoch in tqdm(range(epochs), desc="Training Original AE"):
        x = sample_sparse_batch(cfg.batch_size, p_extra=cfg.sparsity_p, device=device)
        x_hat = orig_ae(x)

        loss = loss_fn(x_hat, x)
        loss.backward()
        opt_ae.step()
        opt_ae.zero_grad()
        
        recon_losses.append(loss.item())
        total_losses.append(loss.item())
        
    print("Original AE training complete.\n")

    print(f'final dense-AE loss: {loss.item():.4e}')

    return orig_ae 

def plot_sae_losses(sae: nn.Module, save_path: Path, recon_losses, faithful_losses, l1_losses, total_losses, faithful_weights):
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))

    ax1.plot(recon_losses, 'b-', alpha=0.7)
    ax1.set_title('Reconstruction Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('MSE Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(faithful_losses, 'r-', alpha=0.7)
    ax2.set_title('Faithful Loss')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('MSE Loss')
    ax2.grid(True, alpha=0.3)

    ax3.plot(l1_losses, 'orange', alpha=0.7)
    ax3.set_title('L1 Sparsity Loss')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('L1 Loss')
    ax3.grid(True, alpha=0.3)

    ax4.plot(total_losses, 'g-', alpha=0.7)
    ax4.set_title('Total Loss')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Total Loss')
    ax4.grid(True, alpha=0.3)

    ax5.plot(faithful_weights, 'purple', alpha=0.7)
    ax5.set_title('Faithful Loss Weight (Ramping)')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Weight')
    ax5.grid(True, alpha=0.3)

    ax6.plot(np.array(recon_losses) / max(recon_losses), 'b-', alpha=0.7, label='Recon (norm)')
    ax6.plot(np.array(faithful_losses) / max(faithful_losses), 'r-', alpha=0.7, label='Faithful (norm)')
    ax6.plot(np.array(l1_losses) / max(l1_losses), 'orange', alpha=0.7, label='L1 (norm)')
    ax6.set_title('All Losses (Normalized)')
    ax6.set_xlabel('Step')
    ax6.set_ylabel('Normalized Loss')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    print(f'Saving losses to {save_path / "figures" / "faithful_sae_training_losses.png"}')
    plt.savefig(save_path / 'figures' / 'faithful_sae_training_losses.png', dpi=300, bbox_inches='tight')
    plt.close()

def train_sae(cfg: Config, orig_ae: nn.Module, device: str):
    sae = Faithful_SAE(input_dim=cfg.input_dim, latent_dim=cfg.concept_dim, hidden_dim=cfg.latent_dim, k=cfg.k, use_topk=cfg.k != 0).to(device)
    sae.train()

    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr_sae)
    
    recon_losses = [] 
    faithful_losses = [] 
    l1_losses = [] 
    total_losses = [] 
    faithful_weights = [] 
    

    best_loss = float('inf')
    pbar = trange(cfg.steps_sae, desc='train Faithful_SAE')

    for step in pbar:
        x = sample_sparse_batch(cfg.batch_size, p_extra=cfg.sparsity_p, device=device)
        
        sae_out, sparse_latent = sae(x)
        sae_encoded = sae.encode(x, use_topk=False)
        
        with torch.no_grad():
            target_out = orig_ae.encode(x)
        
        recon_loss = F.mse_loss(sae_out, target_out)
        W_eff = sae.effective_encoder()
        faithful_loss = F.mse_loss(W_eff, orig_ae.encoder_weights)
        l1_loss = torch.mean(torch.abs(sae_encoded))
        
        progress = step / cfg.steps_sae
        faithful_weight = 2.0 * (progress ** 2)
        
        loss = cfg.lam_recon * recon_loss + faithful_weight * faithful_loss + cfg.lam_l1 * l1_loss
        
        recon_losses.append(recon_loss.item())
        faithful_losses.append(faithful_loss.item())
        l1_losses.append(l1_loss.item())
        total_losses.append(loss.item())
        faithful_weights.append(faithful_weight)
        
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(sae.state_dict(), cfg.save_path / 'checkpoints' / f'best_sae_{cfg.input_dim}_{cfg.latent_dim}_{cfg.concept_dim}_{cfg.k}.pt')
        
        pbar.set_postfix(recon=recon_loss.item(),
                        faithful=faithful_loss.item(),
                        l1=l1_loss.item(),
                        faithful_weight=faithful_weight,
                        best=best_loss)

    print(f'\n✓ training done – best total loss {best_loss:.4e} saved to {cfg.save_path}')

    losses = {
        'recon_losses': recon_losses, 
        'faithful_losses': faithful_losses, 
        'l1_losses': l1_losses, 
        'total_losses': total_losses, 
        'faithful_weights': faithful_weights
    }

    return sae, losses 

def ensure_directories(save_path: Path):
    (save_path / 'figures').mkdir(parents=True, exist_ok=True)
    (save_path / 'checkpoints').mkdir(parents=True, exist_ok=True)

def main():
    cfg = Config()
    set_seed(cfg.seed)
    device = cfg.device
    ensure_directories(cfg.save_path)
    
    print(f'Training AE...')
    orig_ae = train_ae(cfg, device)
    plot_ae_weights(orig_ae, cfg.save_path)

    print(f'Training SAE...')
    sae, losses = train_sae(cfg, orig_ae, device)
    plot_sae_losses(sae, cfg.save_path, losses['recon_losses'], losses['faithful_losses'], losses['l1_losses'], losses['total_losses'], losses['faithful_weights']) 

    print('Saving AE and SAE weights to {cfg.save_path}')
    torch.save(orig_ae.state_dict(), cfg.save_path / 'checkpoints' / f'orig_ae_{cfg.input_dim}_{cfg.latent_dim}.pt')
    torch.save(sae.state_dict(), cfg.save_path / 'checkpoints' / f'sae_{cfg.input_dim}_{cfg.latent_dim}_{cfg.concept_dim}_{cfg.k}_final.pt')

    if sae.effective_encoder().shape[-1] == 2 and orig_ae.encoder_weights.shape[-1] == 2:
        print('Plotting greedy matching...')
        greedy_match_and_plot(orig_ae.encoder_weights, sae.components(), cfg.save_path)

    print('Done!')

import argparse 
if __name__ == '__main__':
    main()