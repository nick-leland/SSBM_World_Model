"""Assemble HDF5 training datasets from parquet state files + rendered frames.

Reads every parquet file in data/states/, joins each row with its rendered
PNG frame (by Slippi frame number), applies frameskip, computes normalization
statistics, and writes two HDF5 files:

    data/hdf5/melee_train.h5
    data/hdf5/melee_val.h5

HDF5 schema (per file):
    pixels      (T, 224, 224, 3)   uint8   — raw RGB frames (ImageNet norm applied at load time)
    action      (T, 13)            float32 — z-score normalised controller inputs
    observation (T, obs_dim)       float32 — z-score normalised game state features
    offsets     (N_episodes + 1,)  int64   — flat-array episode boundaries

Normalization statistics are saved alongside as:
    data/hdf5/norm_stats.json

Usage:
    python preprocess/build_hdf5.py
    python preprocess/build_hdf5.py --states-dir data/states --frames-dir data/frames
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from config import (
    ACTION_COLS,
    FRAMES_DIR,
    HDF5_DIR,
    IMG_SIZE,
    FRAMESKIP,
    RANDOM_SEED,
    STATES_DIR,
    VAL_SPLIT,
    get_obs_cols,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parquet loading
# ---------------------------------------------------------------------------

def find_parquet_pairs(states_dir: Path) -> list[tuple[Path, Path]]:
    """Find all -p1/-p2 parquet pairs.  Returns list of (p1_path, p2_path)."""
    p1_files = sorted(states_dir.glob("*-p1.parquet"))
    pairs = []
    for p1 in p1_files:
        p2 = p1.with_name(p1.name.replace("-p1.parquet", "-p2.parquet"))
        if p2.exists():
            pairs.append((p1, p2))
        else:
            log.warning("No p2 file for %s, skipping pair", p1.name)
    log.info("Found %d replay pairs in %s", len(pairs), states_dir)
    return pairs


def replay_id_from_parquet(parquet_path: Path) -> str:
    """Extract the replay stem used for the frames directory name.

    Parquet files are named  <stage>_<p1>_vs_<p2>_<ts>_<uuid>-p{1,2}.parquet
    The corresponding frames live in data/frames/<stage>_<p1>_vs_<p2>_<ts>_<uuid>/
    We store the original .slp path in the manifest but the parquet name is
    derived from the replay content, so we need to find the matching frame dir
    by stem prefix.
    """
    stem = parquet_path.stem  # e.g. "yoshis_story_fox_vs_falco_2021-01-01_abc12345-p1"
    # Strip the -p1 / -p2 suffix
    if stem.endswith("-p1") or stem.endswith("-p2"):
        stem = stem[:-3]
    return stem


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------

def load_frame(frame_dir: Path, slippi_frame: int) -> np.ndarray | None:
    """Load a single rendered frame as a (H, W, 3) uint8 array resized to IMG_SIZE."""
    frame_path = frame_dir / f"{slippi_frame:06d}.png"
    if not frame_path.exists():
        return None
    img = Image.open(frame_path).convert("RGB").resize(
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR
    )
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Per-episode processing
# ---------------------------------------------------------------------------

def process_episode(
    parquet_path: Path,
    frames_dir: Path,
    obs_cols: list[str],
) -> dict | None:
    """Load one parquet file and its frames, apply frameskip.

    Returns a dict with keys 'frames', 'actions', 'observations' (numpy arrays)
    or None if the episode should be skipped.
    """
    df = pd.read_parquet(parquet_path)

    # Apply frameskip: keep every FRAMESKIP-th row
    df = df.iloc[::FRAMESKIP].reset_index(drop=True)

    if len(df) < 4:  # need at least num_steps frames
        log.warning("Episode too short after frameskip (%d frames), skipping %s",
                    len(df), parquet_path.name)
        return None

    # --- Locate matching frame directory ------------------------------------
    replay_stem = replay_id_from_parquet(parquet_path)
    frame_dir = frames_dir / replay_stem

    has_frames = frame_dir.exists() and any(frame_dir.iterdir())

    # --- Load frames -------------------------------------------------------
    frame_list = []
    valid_mask = np.ones(len(df), dtype=bool)

    for i, slippi_frame in enumerate(df["frame"].values):
        if has_frames:
            arr = load_frame(frame_dir, int(slippi_frame))
        else:
            arr = None

        if arr is None:
            if has_frames:
                # Frame dir exists but this specific frame is missing — drop row
                valid_mask[i] = False
            else:
                # Frame rendering not done yet — use a black placeholder so the
                # pipeline can be tested without Dolphin
                frame_list.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
        else:
            frame_list.append(arr)

    # Filter out rows with missing frames
    df = df[valid_mask].reset_index(drop=True)
    frame_list = [f for f, v in zip(frame_list, valid_mask) if v] if has_frames else frame_list

    if len(df) < 4:
        log.warning("Too few valid frames after filtering for %s, skipping", parquet_path.name)
        return None

    # --- Extract action and observation arrays -----------------------------
    # Coerce bool columns to float32 for uniform dtype
    for col in ACTION_COLS:
        if col in df.columns and df[col].dtype == bool:
            df[col] = df[col].astype("float32")

    missing_action = [c for c in ACTION_COLS if c not in df.columns]
    if missing_action:
        log.warning("Missing action columns %s in %s, skipping", missing_action, parquet_path.name)
        return None

    actions = df[ACTION_COLS].astype("float32").values        # (T, 13)

    obs_present = [c for c in obs_cols if c in df.columns]
    for col in obs_present:
        if df[col].dtype == bool:
            df[col] = df[col].astype("float32")
    observations = df[obs_present].astype("float32").values   # (T, obs_dim)

    frames_arr = np.stack(frame_list, axis=0)                 # (T, H, W, 3)

    assert frames_arr.shape[0] == actions.shape[0] == observations.shape[0], (
        f"Length mismatch: frames={frames_arr.shape[0]}, "
        f"actions={actions.shape[0]}, obs={observations.shape[0]}"
    )

    return {
        "frames": frames_arr,
        "actions": actions,
        "observations": observations,
    }


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def compute_norm_stats(
    episodes: list[dict],
    obs_cols: list[str],
) -> dict:
    """Compute per-column mean and std for action and observation arrays."""
    all_actions = np.concatenate([ep["actions"] for ep in episodes], axis=0)
    all_obs     = np.concatenate([ep["observations"] for ep in episodes], axis=0)

    def _stats(arr: np.ndarray, names: list[str]) -> dict:
        # Compute stats ignoring NaN values
        mean = np.nanmean(arr, axis=0)
        std  = np.nanstd(arr, axis=0)
        std  = np.where(std < 1e-6, 1.0, std)  # avoid division by zero for constant cols
        return {
            "mean": mean.tolist(),
            "std":  std.tolist(),
            "columns": names,
        }

    return {
        "action":      _stats(all_actions, ACTION_COLS),
        "observation": _stats(all_obs,     obs_cols),
    }


def apply_norm(arr: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.array(stats["mean"], dtype=np.float32)
    std  = np.array(stats["std"],  dtype=np.float32)
    return (arr - mean) / std


# ---------------------------------------------------------------------------
# HDF5 writing
# ---------------------------------------------------------------------------

def write_hdf5(
    out_path: Path,
    episodes: list[dict],
    norm_stats: dict,
    obs_dim: int,
) -> None:
    """Write a list of episodes to a single HDF5 file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_frames = sum(ep["frames"].shape[0] for ep in episodes)
    log.info("Writing %s: %d episodes, %d total frames", out_path.name, len(episodes), total_frames)

    with h5py.File(out_path, "w") as f:
        # Pre-allocate datasets with chunking for efficient sequential reads
        chunk_t = min(256, total_frames)
        pixels_ds = f.create_dataset(
            "pixels",
            shape=(total_frames, IMG_SIZE, IMG_SIZE, 3),
            dtype="uint8",
            chunks=(chunk_t, IMG_SIZE, IMG_SIZE, 3),
            compression="lzf",   # fast lossless compression
        )
        action_ds = f.create_dataset(
            "action",
            shape=(total_frames, len(ACTION_COLS)),
            dtype="float32",
            chunks=(chunk_t, len(ACTION_COLS)),
        )
        obs_ds = f.create_dataset(
            "observation",
            shape=(total_frames, obs_dim),
            dtype="float32",
            chunks=(chunk_t, obs_dim),
        )
        offsets_ds = f.create_dataset(
            "offsets",
            shape=(len(episodes) + 1,),
            dtype="int64",
        )

        cursor = 0
        offsets = [0]

        for ep in episodes:
            T = ep["frames"].shape[0]

            # Normalize action and observation
            act_norm = apply_norm(ep["actions"],      norm_stats["action"])
            obs_norm = apply_norm(ep["observations"], norm_stats["observation"])

            # Replace NaN with 0 after normalization (e.g. projectile slots)
            act_norm = np.nan_to_num(act_norm, nan=0.0)
            obs_norm = np.nan_to_num(obs_norm, nan=0.0)

            pixels_ds[cursor:cursor + T] = ep["frames"]
            action_ds[cursor:cursor + T] = act_norm
            obs_ds[cursor:cursor + T]    = obs_norm

            cursor += T
            offsets.append(cursor)

        offsets_ds[:] = np.array(offsets, dtype=np.int64)

        # Store metadata as HDF5 attributes
        f.attrs["action_cols"]  = json.dumps(ACTION_COLS)
        f.attrs["obs_cols"]     = json.dumps(norm_stats["observation"]["columns"])
        f.attrs["frameskip"]    = FRAMESKIP
        f.attrs["img_size"]     = IMG_SIZE
        f.attrs["n_episodes"]   = len(episodes)
        f.attrs["n_frames"]     = total_frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build HDF5 datasets from parquet + rendered frames."
    )
    parser.add_argument("--states-dir", type=Path, default=STATES_DIR)
    parser.add_argument("--frames-dir", type=Path, default=FRAMES_DIR)
    parser.add_argument("--out-dir",    type=Path, default=HDF5_DIR)
    parser.add_argument("--seed",       type=int,  default=RANDOM_SEED)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Discover parquet files --------------------------------------------
    pairs = find_parquet_pairs(args.states_dir)
    if not pairs:
        log.error("No parquet pairs found in %s", args.states_dir)
        sys.exit(1)

    # Flatten pairs into individual episodes (each perspective is one episode)
    all_parquets = [p for pair in pairs for p in pair]

    # --- Determine obs_cols from first available parquet -------------------
    sample_df = pd.read_parquet(all_parquets[0], columns=None)
    obs_cols = get_obs_cols(list(sample_df.columns))
    log.info("action_dim=%d  obs_dim=%d", len(ACTION_COLS), len(obs_cols))

    # --- Train / val split -------------------------------------------------
    rng = random.Random(args.seed)
    rng.shuffle(all_parquets)
    n_val = max(1, int(len(all_parquets) * VAL_SPLIT))
    val_parquets   = all_parquets[:n_val]
    train_parquets = all_parquets[n_val:]
    log.info("Split: %d train, %d val episodes", len(train_parquets), len(val_parquets))

    # --- Load training episodes (need these for norm stats) ----------------
    log.info("Loading training episodes ...")
    train_episodes = []
    for i, pq in enumerate(train_parquets, 1):
        ep = process_episode(pq, args.frames_dir, obs_cols)
        if ep is not None:
            train_episodes.append(ep)
        if i % 100 == 0:
            log.info("  %d / %d loaded (%d kept)", i, len(train_parquets), len(train_episodes))

    if not train_episodes:
        log.error("No valid training episodes.")
        sys.exit(1)

    # --- Compute and save norm stats ---------------------------------------
    log.info("Computing normalization statistics ...")
    norm_stats = compute_norm_stats(train_episodes, obs_cols)
    norm_path = args.out_dir / "norm_stats.json"
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    norm_path.write_text(json.dumps(norm_stats, indent=2))
    log.info("Norm stats saved to %s", norm_path)

    # --- Write train HDF5 --------------------------------------------------
    obs_dim = len(obs_cols)
    write_hdf5(args.out_dir / "melee_train.h5", train_episodes, norm_stats, obs_dim)

    # --- Load and write val HDF5 -------------------------------------------
    log.info("Loading validation episodes ...")
    val_episodes = []
    for pq in val_parquets:
        ep = process_episode(pq, args.frames_dir, obs_cols)
        if ep is not None:
            val_episodes.append(ep)

    if val_episodes:
        write_hdf5(args.out_dir / "melee_val.h5", val_episodes, norm_stats, obs_dim)
    else:
        log.warning("No valid validation episodes.")

    log.info("All done.")


if __name__ == "__main__":
    main()
