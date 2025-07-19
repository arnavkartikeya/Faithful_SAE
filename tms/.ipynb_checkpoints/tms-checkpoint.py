# Testing 5-2 and 40-10 autoencoders on parameter decomposition with greedy and one-to-one matchings 

import sys
import os
import yaml
import random
import numpy as np
from dataclasses import dataclass, asdict
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from tqdm import trange

sys.path.append('../../')

from tqdm import tqdm
from Faithful_SAE.models import Faithful_SAE, ToySuperpositionAE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Faithful_SAE.train_ae import sample_sparse_batch, TrainingConfig, Config

def train_ae(cfg: Config, device: str) -> torch.nn.Module:
    ae = ToySuperpositionAE(cfg.input_dim, cfg.latent_dim).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=cfg.ae_lr)
    loss_fn = nn.MSELoss()

    for _ in tqdm(range(cfg.sae_steps), desc="train dense AE"):
        x = sample_sparse_batch(cfg.batch_size, cfg.input_dim,
                                p_extra=cfg.sparsity_p, device=device)
        loss = loss_fn(ae(x), x)
        opt.zero_grad(); loss.backward(); opt.step()

    print(f"final dense-AE loss: {loss.item():.4e}")
    ae.eval();
    return ae

def train_sae(cfg, ae): 
    sae = Faithful_SAE(cfg.input_dim, cfg.concept_dim, cfg.latent_dim, k=cfg.k).to(cfg.device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.sae_lr)
    best_loss = float('inf')
    pbar = trange(cfg.sae_steps, desc="train Faithful_SAE")
    for _ in pbar:
        
        x = sample_sparse_batch(cfg.batch_size, cfg.input_dim, p_extra=cfg.sparsity_p, device=cfg.device)
        sae_out, sparse_latent = sae(x)
        sae_encoded = sae.encode(x, use_topk=False)

        l1_loss = torch.mean(torch.abs(sae_encoded))
        
        with torch.no_grad():
            target_out = ae.encode(x)
            
        recon_loss = F.mse_loss(sae_out, target_out)
        faithful_loss = F.mse_loss(sae.effective_encoder(), ae.encoder_weights)
        loss = cfg.recon_lam * recon_loss + cfg.faithful_lam * faithful_loss + cfg.l1_lam * l1_loss 
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
        pbar.set_postfix(recon=recon_loss.item(), faithful=faithful_loss.item(), l1=l1_loss.item(), best=best_loss)
    return sae 

def greedy_matching_ae_sae(ae_encoder, sae_components):
    """
    Greedy matching: each AE row can match to any SAE component row.
    """
    ae_encoder_np = ae_encoder.detach().cpu().numpy()
    sae_components_np = sae_components.detach().cpu().numpy()
    
    num_components, num_features, num_hidden = sae_components_np.shape
    
    all_sae_rows = []
    sae_indices = [] 
    
    for c_idx in range(num_components):
        for f_idx in range(num_features):
            all_sae_rows.append(sae_components_np[c_idx, f_idx])
            sae_indices.append((c_idx, f_idx))
    
    all_sae_rows = np.array(all_sae_rows)
    
    matches = []
    max_cosine_sims = []
    
    for ae_row_idx in range(ae_encoder_np.shape[0]):
        ae_row = ae_encoder_np[ae_row_idx].reshape(1, -1)
        
        similarities = cosine_similarity(ae_row, all_sae_rows)[0]
        
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        component_idx, feature_idx = sae_indices[best_match_idx]
        
        matches.append({
            'ae_row_idx': ae_row_idx,
            'sae_component_idx': component_idx,
            'sae_feature_idx': feature_idx,
            'similarity': best_similarity,
            'ae_vector': ae_encoder_np[ae_row_idx],
            'sae_vector': all_sae_rows[best_match_idx]
        })
        
        max_cosine_sims.append(best_similarity)
    
    mean_max_cosine_sim = np.mean(max_cosine_sims)
    
    return mean_max_cosine_sim, matches

def hungarian_matching_ae_sae(ae_encoder, sae_components):
    """
    Hungarian matching: each AE row matches to a unique SAE component (one-to-one on components).
    """
    ae_encoder_np = ae_encoder.detach().cpu().numpy()
    sae_components_np = sae_components.detach().cpu().numpy()
    
    num_components, num_features, num_hidden = sae_components_np.shape
    num_ae_features = ae_encoder_np.shape[0]
    
    component_best_similarities = np.zeros((num_ae_features, num_components))
    component_best_feature_indices = np.zeros((num_ae_features, num_components), dtype=int)
    
    for ae_row_idx in range(num_ae_features):
        ae_row = ae_encoder_np[ae_row_idx].reshape(1, -1)
        
        for c_idx in range(num_components):
            component_rows = sae_components_np[c_idx]  
            
            similarities = cosine_similarity(ae_row, component_rows)[0]
            
            best_feature_idx = np.argmax(similarities)
            best_similarity = similarities[best_feature_idx]
            
            component_best_similarities[ae_row_idx, c_idx] = best_similarity
            component_best_feature_indices[ae_row_idx, c_idx] = best_feature_idx
    
    cost_matrix = -component_best_similarities
    
    if num_components > num_ae_features:
        ae_indices, component_indices = linear_sum_assignment(cost_matrix)
    else:
        ae_indices, component_indices = linear_sum_assignment(cost_matrix[:, :min(num_components, num_ae_features)])
    
    matches = []
    max_cosine_sims = []
    
    for ae_idx, comp_idx in zip(ae_indices, component_indices):
        best_similarity = component_best_similarities[ae_idx, comp_idx]
        best_feature_idx = component_best_feature_indices[ae_idx, comp_idx]
        
        matches.append({
            'ae_row_idx': ae_idx,
            'sae_component_idx': comp_idx,
            'sae_feature_idx': best_feature_idx,
            'similarity': best_similarity,
            'ae_vector': ae_encoder_np[ae_idx],
            'sae_vector': sae_components_np[comp_idx, best_feature_idx]
        })
        
        max_cosine_sims.append(best_similarity)
    
    mean_max_cosine_sim = np.mean(max_cosine_sims)
    
    return mean_max_cosine_sim, matches

def print_matching_results(greedy_score, greedy_matches, hungarian_score, hungarian_matches):
    print("=" * 80)
    print("AE vs SAE Matching Results")
    print("=" * 80)
    
    print(f"\n🎯 GREEDY MATCHING RESULTS:")
    print(f"   Mean Max Cosine Similarity: {greedy_score:.6f}")
    print(f"   Number of matches: {len(greedy_matches)}")
    
    print(f"\n🎯 HUNGARIAN MATCHING RESULTS (Component-Constrained):")
    print(f"   Mean Max Cosine Similarity: {hungarian_score:.6f}")
    print(f"   Number of matches: {len(hungarian_matches)}")
    
    print(f"\n📊 COMPARISON:")
    print(f"   Difference: {greedy_score - hungarian_score:.6f}")
    print(f"   Relative improvement (Greedy vs Hungarian): {((greedy_score - hungarian_score) / hungarian_score * 100):.2f}%")
    
    print(f"\n📈 COMPONENT USAGE ANALYSIS:")
    
    greedy_component_usage = {}
    for match in greedy_matches:
        comp_idx = match['sae_component_idx']
        greedy_component_usage[comp_idx] = greedy_component_usage.get(comp_idx, 0) + 1
    
    print(f"   Greedy - Unique components used: {len(greedy_component_usage)}/512")
    print(f"   Greedy - Max uses of single component: {max(greedy_component_usage.values())}")
    
    hungarian_component_usage = {}
    for match in hungarian_matches:
        comp_idx = match['sae_component_idx']
        hungarian_component_usage[comp_idx] = hungarian_component_usage.get(comp_idx, 0) + 1
    
    print(f"   Hungarian - Unique components used: {len(hungarian_component_usage)}/512")
    print(f"   Hungarian - Max uses of single component: {max(hungarian_component_usage.values())}")
    
    greedy_sims = [match['similarity'] for match in greedy_matches]
    hungarian_sims = [match['similarity'] for match in hungarian_matches]
    
    print(f"\n📊 SIMILARITY STATISTICS:")
    print(f"   Greedy - Min: {min(greedy_sims):.4f}, Max: {max(greedy_sims):.4f}, Std: {np.std(greedy_sims):.4f}")
    print(f"   Hungarian - Min: {min(hungarian_sims):.4f}, Max: {max(hungarian_sims):.4f}, Std: {np.std(hungarian_sims):.4f}")

if __name__ == '__main__': 
    cfg_40_10 = TrainingConfig(k=1) 
    cfg_5_2 = TrainingConfig(concepts=64, concept_dim=64, input_dim=5, latent_dim=2, k=1) 


    ae_40_10 = train_ae(cfg_40_10, cfg_40_10.device)
    ae_5_2 = train_ae(cfg_5_2, cfg_5_2.device) 

    real_sae_40_10 = train_sae(cfg_40_10, ae_40_10)
    real_sae_5_2 = train_sae(cfg_5_2, ae_5_2)

    print('='*100)
    print("Results for 40-10 encoder...")
    print('='*100)

    print("Computing Greedy Matching...")
    greedy_score_40_10, greedy_matches_40_10 = greedy_matching_ae_sae(ae_40_10.encoder_weights, real_sae_40_10.components())

    print("Computing Hungarian Matching...")
    hungarian_score_40_10, hungarian_matches_40_10 = hungarian_matching_ae_sae(ae_40_10.encoder_weights, real_sae_40_10.components())

    print_matching_results(greedy_score_40_10, greedy_matches_40_10, hungarian_score_40_10, hungarian_matches_40_10)

    print('='*100)
    print("Results for 5-2 encoder...")
    print('='*100)

    print("Computing Greedy Matching...")
    greedy_score_5_2, greedy_matches_5_2 = greedy_matching_ae_sae(ae_5_2.encoder_weights, real_sae_5_2.components())

    print("Computing Hungarian Matching...")
    hungarian_score_5_2, hungarian_matches_5_2 = hungarian_matching_ae_sae(ae_5_2.encoder_weights, real_sae_5_2.components())

    print_matching_results(greedy_score_5_2, greedy_matches_5_2, hungarian_score_5_2, hungarian_matches_5_2)

    

    
    


