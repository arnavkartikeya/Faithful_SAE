# train_sae.py
import argparse, yaml, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime
from tqdm import tqdm, trange
import random, numpy as np

from models import ToySuperpositionAE, Faithful_SAE
from train_ae import sample_sparse_batch

@dataclass
class Config:
    create_dir: bool = True
    input_dim: int = 40
    latent_dim: int = 10
    concept_dim: int = 40
    k: int = 1
    batch_size: int = 2048
    sparsity_p: float = 0.05
    epochs_ae: int = 5_000
    steps_sae: int = 30_000
    lr_ae: float = 1e-3
    lr_sae: float = 1e-3
    lam_recon: float = 10.0
    lam_faith: float = 1.0
    seed: int | None = None
    out_dir: str = "runs"
    run_name: str = field(init=False)
    save_dir: Path = field(init=False)

    def __post_init__(self):
        if self.seed is None:
            self.seed = random.randint(0, 2**32 - 1)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        dims = f"in{self.input_dim}_c{self.concept_dim}_lat{self.latent_dim}"
        self.run_name = f"{ts}_{dims}_seed{self.seed}"
        self.save_dir = Path(self.out_dir) / self.run_name
        if self.create_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_ae(cfg: Config, device: str):
    ae = ToySuperpositionAE(cfg.input_dim, cfg.latent_dim).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=cfg.lr_ae)
    loss_fn = nn.MSELoss()
    for _ in tqdm(range(cfg.epochs_ae), desc="train dense AE"):
        x = sample_sparse_batch(cfg.batch_size, cfg.input_dim, p_extra=cfg.sparsity_p, device=device)
        loss = loss_fn(ae(x), x)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.save(ae.state_dict(), cfg.save_dir / "dense_ae.pt")
    ae.eval(); ae.cpu()
    return ae

def train_sae(cfg: Config, ae: nn.Module, device: str):
    sae = Faithful_SAE(cfg.input_dim, cfg.concept_dim, cfg.latent_dim, k=cfg.k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr_sae)
    best_loss = float('inf')
    pbar = trange(cfg.steps_sae, desc="train Faithful_SAE")
    for _ in pbar:
        x = sample_sparse_batch(cfg.batch_size, cfg.input_dim, p_extra=cfg.sparsity_p, device=device)
        sae_out = sae(x)
        with torch.no_grad():
            target_out = ae(x)
        recon_loss = F.mse_loss(sae_out, target_out)
        faithful_loss = F.mse_loss(sae.effective_encoder(), ae.encoder_weights)
        loss = cfg.lam_recon * recon_loss + cfg.lam_faith * faithful_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(sae.state_dict(), cfg.save_dir / "faithful_sae.pt")
        pbar.set_postfix(recon=recon_loss.item(), faithful=faithful_loss.item(), best=best_loss)

def main():
    parser = argparse.ArgumentParser()
    for f in Config.__dataclass_fields__.values():
        if f.init:
            default = None if f.name == "seed" else f.default
            parser.add_argument(f"--{f.name}", type=type(f.default), default=default)
    cfg = Config(**vars(parser.parse_args()))
    set_seed(cfg.seed)
    def cast_paths(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, Path):
                out[k] = str(v)
            elif isinstance(v, dict):
                out[k] = cast_paths(v)
            else:
                out[k] = v
        return out
    with open(cfg.save_dir / "config.yaml", "w") as fh:
        yaml.safe_dump(cast_paths(asdict(cfg)), fh)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ae = train_ae(cfg, device)
    train_sae(cfg, ae.to(device), device)

if __name__ == "__main__":
    main()
