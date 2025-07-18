import argparse, yaml, torch, torch.nn as nn
from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm import tqdm
from models import ToySuperpositionAE

@dataclass
class Config:
    input_dim:     int   = 40
    latent_dim:    int   = 10        
    batch_size:    int   = 2048
    sparsity_p:    float = 0.05
    epochs:        int   = 5_000
    lr_ae:         float = 1e-3
    save_ae_path:  str   = "dense_ae.pt"

    lr_sae:   float = 1e-3
    steps_sae:int  = 30_000
    lam_recon:float= 10.0
    lam_faith:float= 1.0
    save_sae_path:str = "faithful_sae_best.pt"

@torch.no_grad()
def sample_sparse_batch(batch_size: int,
                        input_dim: int,
                        p_extra: float = 0.05,
                        device: str = "cuda") -> torch.Tensor:
    chosen = torch.randint(0, input_dim, (batch_size, 1), device=device)      # (B,1)

    vals = torch.rand(batch_size, input_dim, device=device)                   # (B,D)

    mask = torch.rand(batch_size, input_dim, device=device) < p_extra         # extras
    # mask.scatter_(1, chosen, True)                                            # force the 1

    return vals * mask.float()                                                # (B,D)


def train_ae(cfg: Config, device: str) -> torch.nn.Module:
    ae = ToySuperpositionAE(cfg.input_dim, cfg.latent_dim).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=cfg.lr_ae)
    loss_fn = nn.MSELoss()

    for _ in tqdm(range(cfg.epochs), desc="train dense AE"):
        x = sample_sparse_batch(cfg.batch_size, cfg.input_dim,
                                p_extra=cfg.sparsity_p, device=device)
        loss = loss_fn(ae(x), x)
        opt.zero_grad(); loss.backward(); opt.step()

    print(f"final dense-AE loss: {loss.item():.4e}")
    ae.eval();  ae.cpu()
    torch.save(ae.state_dict(), cfg.save_ae_path)
    print(f"✓ saved weights to {cfg.save_ae_path}")
    return ae

def main():
    parser = argparse.ArgumentParser()
    for f in Config.__dataclass_fields__.values():
        parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    args = parser.parse_args()
    cfg = Config(**vars(args))

    cfg_path = Path(cfg.save_ae_path).with_suffix(".yaml")
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(asdict(cfg), fh)
    print(f"✓ saved config to {cfg_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _ = train_ae(cfg, device)

if __name__ == "__main__":
    main()
