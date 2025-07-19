import yaml, torch, random, numpy as np
from pathlib import Path
from dataclasses import fields
from models import ToySuperpositionAE, Faithful_SAE       # <- your modules
from train_sae import Config, set_seed                    # <- same dataclass

def _yaml_to_cfg_dict(path: Path) -> dict:
    """Parse config.yaml, convert *_path/_dir strings → Path objects."""
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    for k, v in raw.items():
        if isinstance(v, str) and (k.endswith("_path") or k.endswith("_dir")):
            raw[k] = Path(v)
    return raw

def load_run(run_folder: str | Path, device: str = "cpu"):
    """
    Returns (cfg, orig_ae, sae) loaded from a training run directory.
    ``run_folder`` must contain:
        – config.yaml
        – dense_ae.pt
        – faithful_sae.pt
    """
    run_folder = Path(run_folder)
    cfg_dict   = _yaml_to_cfg_dict(run_folder / "config.yaml")

    valid = {f.name for f in fields(Config) if f.init}
    # cfg_init = {k: v for k, v in cfg_dict.items() if k in valid}
    # cfg = Config(create_dir=False, **cfg_init)
    cfg = Config(create_dir=False, **{k: v for k, v in cfg_dict.items() if k in valid and k != "create_dir"})


    cfg.run_name = cfg_dict.get("run_name", run_folder.name)
    cfg.save_dir = run_folder

    set_seed(cfg.seed)

    orig_ae = ToySuperpositionAE(cfg.input_dim, cfg.latent_dim).to(device)
    orig_ae.load_state_dict(torch.load(run_folder / "dense_ae.pt",
                                       map_location=device))
    orig_ae.eval()

    sae = Faithful_SAE(cfg.input_dim, cfg.concept_dim, cfg.latent_dim,
                       k=cfg.k).to(device)
    sae.load_state_dict(torch.load(run_folder / "faithful_sae.pt",
                                   map_location=device))
    sae.eval()

    print(f"✓ restored run {cfg.run_name}  (seed={cfg.seed})")
    return cfg, orig_ae, sae
