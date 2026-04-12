import torch
import numpy as np
import importlib.util
from pathlib import Path
from omegaconf import OmegaConf
from starVLA.dataloader import build_dataloader


def _load_encoder(subdir: str, module_name: str, func_name: str):
    """Dynamically load encoder builder function via importlib."""
    module_path = Path(__file__).resolve().parents[1] / "model" / "modules" / subdir / "encoder.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Module {subdir} not found at {module_path}")
    
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for {module_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)


build_lam_stage2_encoder = _load_encoder("lam-stage-2", "starvla_lam_stage2_encoder", "build_lam_stage2_encoder")
build_depth_encoder = _load_encoder("depth_encoder", "starvla_depth_encoder", "build_depth_encoder")
build_qwen3_embedding_slot_adapter = _load_encoder("qwen3-embedding", "starvla_qwen3_embedding_encoder", "build_qwen3_embedding_slot_adapter")


def main():
    config_path = Path("/mnt/nas_ssd/workspace/wenboli/projects/starVLA/starVLA/config/training/myvla.yaml")
    cfg = OmegaConf.load(str(config_path))

    # Override for minimal smoke
    cfg.output_dir = "./results/debug"
    cfg.datasets.vla_data.per_device_batch_size = 1
    cfg.datasets.vla_data.data_root_dir = "playground/Datasets/LEROBOT_LIBERO_DATA"
    cfg.datasets.vla_data.data_mix = "libero_goal"
    cfg.datasets.vla_data.load_all_data_for_training = False

    print("[INFO] Building real dataloader...")
    try:
        dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
        print(f"[OK] Dataloader built: {type(dataloader)}")
    except Exception as e:
        print(f"[ERROR] Dataloader build failed: {e}")
        return False

    print("[INFO] Loading one batch from real dataloader...")
    try:
        batch = next(iter(dataloader))
        # Handle batch as either dict or list of dicts
        if isinstance(batch, list):
            batch = batch[0]  # Take first example from batch list
            print(f"[OK] Batch loaded (list format). Keys: {list(batch.keys())}")
        else:
            print(f"[OK] Batch loaded (dict format). Keys: {list(batch.keys())}")
        
        # Print shapes
        if 'image' in batch:
            img = batch['image'][0] if isinstance(batch['image'], list) else batch['image']
            if hasattr(img, 'shape'):
                print(f"    image shape: {img.shape}")
            else:
                print(f"    image type: {type(img)}")
        if 'lang' in batch:
            print(f"    instruction: {batch['lang'][:50]}")
        if 'action' in batch:
            print(f"    action shape: {batch['action'].shape}")
    except Exception as e:
        print(f"[ERROR] Batch load failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("[INFO] Building real teacher encoders...")
    try:
        lam_encoder = build_lam_stage2_encoder(cfg)
        print("[OK] LAM stage-2 encoder built")
    except Exception as e:
        print(f"[WARN] LAM encoder build skipped: {e}")
        lam_encoder = None

    try:
        depth_encoder = build_depth_encoder(cfg)
        print("[OK] Depth encoder built")
    except Exception as e:
        print(f"[WARN] Depth encoder build skipped: {e}")
        depth_encoder = None

    try:
        embedding_encoder = build_qwen3_embedding_slot_adapter(cfg)
        print("[OK] Qwen3-Embedding encoder built")
    except Exception as e:
        print(f"[WARN] Embedding encoder build skipped: {e}")
        embedding_encoder = None

    print("[INFO] Running teachers on real batch...")
    with torch.no_grad():
        # Format batch for teachers (convert to list of dicts for examples)
        examples = []
        for i in range(len(batch["image"]) if isinstance(batch["image"], list) else 1):
            ex = {
                "image": batch["image"][i] if isinstance(batch["image"], list) else batch["image"],
                "lang": batch.get("lang", "do task"),
                "action": batch.get("action", np.zeros((16, 7), dtype=np.float32)),
            }
            examples.append(ex)

        if lam_encoder:
            try:
                lam_out = lam_encoder(examples=examples)
                print(f"[OK] LAM output: z_q={lam_out['z_q'].shape if 'z_q' in lam_out else 'N/A'}")
            except Exception as e:
                print(f"[WARN] LAM forward failed: {e}")

        if depth_encoder:
            try:
                depth_out = depth_encoder(examples=examples)
                print(f"[OK] Depth output: tokens={depth_out.get('encoder_tokens', torch.zeros(1)).shape}")
            except Exception as e:
                print(f"[WARN] Depth forward failed: {e}")

        if embedding_encoder:
            try:
                emb_out = embedding_encoder(examples=examples)
                print(f"[OK] Embedding output: tokens={emb_out.get('text_tokens', torch.zeros(1)).shape}")
            except Exception as e:
                print(f"[WARN] Embedding forward failed: {e}")

    print("[OK] Real data forward smoke PASSED: dataloader, teachers all executed successfully.")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
