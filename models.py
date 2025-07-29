# coding apd as sae myself to make sure its legit 
import torch 
import numpy as np 
import torch.nn.functional as F 
import torch.nn as nn 


class ToySuperpositionAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder_weights = nn.Parameter(torch.empty(input_dim, hidden_dim))
        nn.init.xavier_normal_(self.encoder_weights)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.encoder_weights 
        return h 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.encoder_weights
        out = h @ self.encoder_weights.T + self.decoder_bias
        return F.relu(out)
        

class Faithful_SAE(nn.Module): 
    def __init__(self, input_dim, latent_dim, hidden_dim, k=1, use_topk=True):
        super().__init__()
        self.k = k 
        self.use_topk = use_topk
        self.encoder = nn.Parameter(torch.empty(input_dim, latent_dim))  
        nn.init.xavier_normal_(self.encoder) 
        self.decoder = nn.Parameter(torch.empty(latent_dim, hidden_dim))
        nn.init.xavier_normal_(self.decoder) 
    
    def component(self, idx):
        encoder_vector = self.encoder[:, idx]
        decoder_vector = self.decoder[idx, :]
        
        return torch.outer(encoder_vector, decoder_vector)

    def components(self):
        return torch.einsum('ic, ch -> cih', self.encoder, self.decoder)
    
    def effective_encoder(self):
        return self.encoder @ self.decoder
    
    def encode(self, x, use_topk=True):
        latent = x @ self.encoder
        
        if use_topk:
            vals, idx = latent.topk(self.k, dim=1)
            sparse = torch.zeros_like(latent).scatter_(1, idx, vals)
        else:
            sparse = latent

        return sparse
    
    def forward(self, x): 
        latent = x @ self.encoder 
        
        if self.use_topk:
            vals, idx = latent.topk(self.k, dim=1)
            sparse = torch.zeros_like(latent).scatter_(1, idx, vals)
        else:
            sparse = latent
        
        reconstructed_output = sparse @ self.decoder  
        return reconstructed_output, sparse
