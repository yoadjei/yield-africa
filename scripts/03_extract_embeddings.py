"""
Phase 3 — Extract patch embeddings using two foundation models.

Models:
  - Prithvi-EO-1.0-100M  (ibm-nasa-geospatial/Prithvi-EO-1.0-100M)
    Geospatial-specific MAE pretrained on HLS (Sentinel-2 + Landsat).
    Loaded directly from custom prithvi_mae.py + local weights (not AutoModel).
    Input: 6 bands (Blue, Green, Red, NIR-narrow, SWIR-1, SWIR-2), 224x224.
    Embedding: encoder CLS token, 768-dim.

  - ViT-Base  (google/vit-base-patch16-224)
    General-purpose Vision Transformer pretrained on ImageNet-21k.
    Baseline comparison: geospatial vs general vision pretraining.
    Embedding: CLS token, 768-dim.

Paper narrative: comparing domain-specific vs general vision pretraining for
yield estimation. Both produce 768-dim CLS-token embeddings.

Device priority: XPU (Intel Arc) -> CUDA -> CPU

Resume logic: checkpoints written every CHECKPOINT_EVERY batches.
  Intermediate: data/processed/embeddings_prithvi_ckpt.parquet
  Final:        data/processed/embeddings_prithvi.parquet
  If final exists, extraction is skipped entirely.
  If checkpoint exists, already-done field_ids are skipped.

Run:
    python scripts/03_extract_embeddings.py --model prithvi
    python scripts/03_extract_embeddings.py --model vit
    python scripts/03_extract_embeddings.py --model all   (default)

Outputs:
    data/processed/embeddings_prithvi.parquet
    data/processed/embeddings_vit.parquet
"""

import argparse
import sys
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from transformers import ViTModel

# local Prithvi architecture (downloaded from HF repo)
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from prithvi_mae import PrithviMAE

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PATCHES_DIR = RAW_DIR / "s2_patches"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# band order in GeoTIFFs — must match 01b_gee_extract.py S2_BANDS
BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# Prithvi expects 6 HLS bands: Blue, Green, Red, NIR-narrow, SWIR-1, SWIR-2
PRITHVI_BAND_IDX = [
    BAND_NAMES.index("B2"),    # Blue  → HLS B02
    BAND_NAMES.index("B3"),    # Green → HLS B03
    BAND_NAMES.index("B4"),    # Red   → HLS B04
    BAND_NAMES.index("B8A"),   # NIR narrow → HLS B05
    BAND_NAMES.index("B11"),   # SWIR 1     → HLS B06
    BAND_NAMES.index("B12"),   # SWIR 2     → HLS B07
]

# Prithvi stats from config.yaml, scaled to [0,1] range (raw / 10000)
PRITHVI_MEAN = np.array([775.23, 1080.99, 1228.59, 2497.20, 2204.21, 1610.83],
                         dtype=np.float32) / 10000.0
PRITHVI_STD  = np.array([1281.53, 1270.03, 1399.48, 1368.34, 1291.68, 1154.51],
                         dtype=np.float32) / 10000.0

# Prithvi model architecture args from config.yaml
PRITHVI_MODEL_ARGS = dict(
    img_size=224,
    in_chans=6,
    num_frames=1,       # we have single timestep per patch
    patch_size=16,
    tubelet_size=1,
    embed_dim=768,
    depth=12,
    num_heads=12,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
)

TARGET_SIZE = 224        # both models expect 224x224
CHECKPOINT_EVERY = 50   # write checkpoint every N batches


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.xpu.is_available():
        print("Device: Intel Arc XPU")
        return torch.device("xpu")
    elif torch.cuda.is_available():
        print("Device: CUDA GPU")
        return torch.device("cuda")
    else:
        print("Device: CPU")
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: Path) -> set:
    """Return set of field_ids already done in checkpoint."""
    if not ckpt_path.exists():
        return set()
    df = pd.read_parquet(ckpt_path, columns=["field_id"])
    done = set(df["field_id"].tolist())
    print(f"  Resuming from checkpoint: {len(done):,} field_ids already done")
    return done


def save_checkpoint(records: list, ckpt_path: Path):
    """Overwrite checkpoint parquet with full records list."""
    pd.DataFrame(records).to_parquet(ckpt_path, index=False)


# ---------------------------------------------------------------------------
# Patch loading
# ---------------------------------------------------------------------------

def load_patch(patch_path: Path) -> np.ndarray | None:
    """
    Load GeoTIFF as float32 array (C, H, W), scaled to [0, 1].
    Returns None if unreadable.
    """
    try:
        with rasterio.open(patch_path) as src:
            data = src.read().astype(np.float32)
        data = data / 10000.0
        # fill nodata (<=0) with band median
        for c in range(data.shape[0]):
            band = data[c]
            valid = band[band > 0]
            fill = float(np.median(valid)) if len(valid) > 0 else 0.0
            data[c] = np.where(band <= 0, fill, band)
        return data
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Prithvi-EO-1.0-100M  (loaded from local weights via custom class)
# ---------------------------------------------------------------------------

def load_prithvi(device: torch.device):
    weights_path = MODELS_DIR / "Prithvi_EO_V1_100M.pt"
    if not weights_path.exists():
        sys.exit(
            f"Prithvi weights not found at {weights_path}.\n"
            "Run: python -c \"from huggingface_hub import hf_hub_download; import shutil; "
            "shutil.copy(hf_hub_download('ibm-nasa-geospatial/Prithvi-EO-1.0-100M', "
            "'Prithvi_EO_V1_100M.pt'), 'models/Prithvi_EO_V1_100M.pt')\""
        )
    print("Loading Prithvi-EO-1.0-100M from local weights...")
    model = PrithviMAE(**PRITHVI_MODEL_ARGS)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    # drop fixed pos_embed (re-computed from grid at runtime)
    for k in list(state_dict.keys()):
        if "pos_embed" in k:
            del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    print(f"  Loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return model


def prithvi_preprocess(data: np.ndarray) -> torch.Tensor:
    """
    Select 6 Prithvi bands, normalise, resize to 224x224.
    Returns tensor (6, 224, 224).
    """
    bands = data[PRITHVI_BAND_IDX]   # (6, H, W)
    mean = PRITHVI_MEAN[:, None, None]
    std  = PRITHVI_STD[:, None, None]
    bands = (bands - mean) / (std + 1e-8)
    t = torch.from_numpy(bands)
    if t.shape[1] != TARGET_SIZE or t.shape[2] != TARGET_SIZE:
        t = F.interpolate(t.unsqueeze(0), size=(TARGET_SIZE, TARGET_SIZE),
                          mode="bilinear", align_corners=False).squeeze(0)
    return t


def extract_prithvi(df: pd.DataFrame, model, device: torch.device,
                    batch_size: int = 16,
                    ckpt_path: Path = None) -> pd.DataFrame:
    done_ids = load_checkpoint(ckpt_path) if ckpt_path else set()
    todo = df[~df["field_id"].isin(done_ids)] if done_ids else df

    records = []
    if ckpt_path and ckpt_path.exists():
        records = pd.read_parquet(ckpt_path).to_dict("records")

    failed = 0
    batches = [todo.iloc[i:i+batch_size] for i in range(0, len(todo), batch_size)]
    print(f"  Prithvi: {len(done_ids):,} already done, {len(todo):,} remaining")

    for batch_num, batch_df in enumerate(tqdm(batches, desc="Prithvi")):
        tensors, ids = [], []
        for _, row in batch_df.iterrows():
            path = PATCHES_DIR / row["country"] / f"{row['field_id']}_{int(row['year'])}.tif"
            data = load_patch(path)
            if data is None:
                failed += 1
                continue
            tensors.append(prithvi_preprocess(data))
            ids.append(row["field_id"])

        if not tensors:
            continue

        # Prithvi input: (B, C, T, H, W) — T=1
        batch = torch.stack(tensors).unsqueeze(2).to(device)   # (B, 6, 1, 224, 224)

        with torch.no_grad():
            try:
                features = model.forward_features(batch)
                # features is list of block outputs; last is normed encoder output
                # shape: (B, 1 + num_patches, 768) — token 0 is CLS
                enc_out = features[-1]   # (B, N+1, 768)
                emb = enc_out[:, 0, :]   # CLS token, (B, 768)
                emb = emb.cpu().float().numpy()
            except Exception as e:
                print(f"\n  Prithvi batch error: {e}")
                failed += len(ids)
                continue

        for i, fid in enumerate(ids):
            records.append({"field_id": fid,
                            **{f"prithvi_{j}": float(emb[i, j])
                               for j in range(emb.shape[1])}})

        if ckpt_path and (batch_num + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(records, ckpt_path)
            tqdm.write(f"  checkpoint saved ({len(records):,} done)")

    if ckpt_path:
        save_checkpoint(records, ckpt_path)

    print(f"  Prithvi: {len(records):,} extracted total, {failed} failed")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# ViT-Base (general vision baseline)
# ---------------------------------------------------------------------------

def load_vit(device: torch.device):
    print("Loading ViT-Base (google/vit-base-patch16-224)...")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    model = model.to(device).eval()
    print(f"  Loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return model


# ViT ImageNet normalisation — applied to RGB (B4, B3, B2)
VIT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
VIT_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

# band indices for RGB from our 10-band order
RGB_IDX = [
    BAND_NAMES.index("B4"),   # Red
    BAND_NAMES.index("B3"),   # Green
    BAND_NAMES.index("B2"),   # Blue
]


def vit_preprocess(data: np.ndarray) -> torch.Tensor:
    """
    Extract RGB bands, normalise with ImageNet stats, resize to 224x224.
    Returns tensor (3, 224, 224).
    """
    rgb = data[RGB_IDX]   # (3, H, W)
    rgb = np.clip(rgb, 0, 1)
    rgb = (rgb - VIT_MEAN) / (VIT_STD + 1e-8)
    t = torch.from_numpy(rgb)
    if t.shape[1] != TARGET_SIZE or t.shape[2] != TARGET_SIZE:
        t = F.interpolate(t.unsqueeze(0), size=(TARGET_SIZE, TARGET_SIZE),
                          mode="bilinear", align_corners=False).squeeze(0)
    return t


def extract_vit(df: pd.DataFrame, model, device: torch.device,
                batch_size: int = 16,
                ckpt_path: Path = None) -> pd.DataFrame:
    done_ids = load_checkpoint(ckpt_path) if ckpt_path else set()
    todo = df[~df["field_id"].isin(done_ids)] if done_ids else df

    records = []
    if ckpt_path and ckpt_path.exists():
        records = pd.read_parquet(ckpt_path).to_dict("records")

    failed = 0
    batches = [todo.iloc[i:i+batch_size] for i in range(0, len(todo), batch_size)]
    print(f"  ViT: {len(done_ids):,} already done, {len(todo):,} remaining")

    for batch_num, batch_df in enumerate(tqdm(batches, desc="ViT")):
        tensors, ids = [], []
        for _, row in batch_df.iterrows():
            path = PATCHES_DIR / row["country"] / f"{row['field_id']}_{int(row['year'])}.tif"
            data = load_patch(path)
            if data is None:
                failed += 1
                continue
            tensors.append(vit_preprocess(data))
            ids.append(row["field_id"])

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)   # (B, 3, 224, 224)

        with torch.no_grad():
            try:
                out = model(pixel_values=batch)
                emb = out.last_hidden_state[:, 0, :]   # CLS token, (B, 768)
                emb = emb.cpu().float().numpy()
            except Exception as e:
                print(f"\n  ViT batch error: {e}")
                failed += len(ids)
                continue

        for i, fid in enumerate(ids):
            records.append({"field_id": fid,
                            **{f"vit_{j}": float(emb[i, j])
                               for j in range(emb.shape[1])}})

        if ckpt_path and (batch_num + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(records, ckpt_path)
            tqdm.write(f"  checkpoint saved ({len(records):,} done)")

    if ckpt_path:
        save_checkpoint(records, ckpt_path)

    print(f"  ViT: {len(records):,} extracted total, {failed} failed")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["prithvi", "vit", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (reduce to 8 if OOM)")
    args = parser.parse_args()

    master_path = PROCESSED_DIR / "master_dataset.parquet"
    if not master_path.exists():
        sys.exit(f"Run 02_preprocess.py first. Expected: {master_path}")

    df = pd.read_parquet(master_path)
    print(f"Loaded {len(df):,} observations from master_dataset.parquet")

    device = get_device()
    bs = args.batch_size

    run_prithvi = args.model in ("prithvi", "all")
    run_vit     = args.model in ("vit", "all")

    if run_prithvi:
        out_path  = PROCESSED_DIR / "embeddings_prithvi.parquet"
        ckpt_path = PROCESSED_DIR / "embeddings_prithvi_ckpt.parquet"
        if out_path.exists():
            print("Prithvi embeddings exist — skipping. Delete to re-extract.")
        else:
            model = load_prithvi(device)
            emb_df = extract_prithvi(df, model, device, batch_size=bs, ckpt_path=ckpt_path)
            if len(emb_df) > 0:
                emb_df.to_parquet(out_path, index=False)
                print(f"Saved: {out_path}  ({len(emb_df):,} rows, {emb_df.shape[1]-1} dims)")
                if ckpt_path.exists():
                    ckpt_path.unlink()
            del model
            if device.type == "xpu":
                torch.xpu.empty_cache()

    if run_vit:
        out_path  = PROCESSED_DIR / "embeddings_vit.parquet"
        ckpt_path = PROCESSED_DIR / "embeddings_vit_ckpt.parquet"
        if out_path.exists():
            print("ViT embeddings exist — skipping. Delete to re-extract.")
        else:
            model = load_vit(device)
            emb_df = extract_vit(df, model, device, batch_size=bs, ckpt_path=ckpt_path)
            if len(emb_df) > 0:
                emb_df.to_parquet(out_path, index=False)
                print(f"Saved: {out_path}  ({len(emb_df):,} rows, {emb_df.shape[1]-1} dims)")
                if ckpt_path.exists():
                    ckpt_path.unlink()
            del model
            if device.type == "xpu":
                torch.xpu.empty_cache()

    print("\nDone. Next: python scripts/04_train_eval.py")


if __name__ == "__main__":
    main()
