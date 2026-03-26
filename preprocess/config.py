"""Shared configuration for the Melee preprocessing pipeline."""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

RAW_DIR    = DATA_ROOT / "raw"     # downloaded .slp files
STATES_DIR = DATA_ROOT / "states"  # parquet files (game state per frame)
FRAMES_DIR = DATA_ROOT / "frames"  # rendered PNGs from Dolphin, one subdir per replay
HDF5_DIR   = DATA_ROOT / "hdf5"    # final HDF5 datasets for le-wm

# Path to the slippi-frame-extractor repo (imported at runtime via sys.path)
EXTRACTOR_DIR = PROJECT_ROOT / "slippi-frame-extractor"

# ---------------------------------------------------------------------------
# HuggingFace dataset
# ---------------------------------------------------------------------------
HF_DATASET_ID = "erickfm/slippi-public-dataset-v3.7"

# ---------------------------------------------------------------------------
# HDF5 / training parameters  (must stay in sync with le-wm config)
# ---------------------------------------------------------------------------
IMG_SIZE   = 224   # pixels are stored raw uint8; le-wm resizes if needed
FRAMESKIP  = 5     # subsample every Nth game frame (60 fps → 12 effective fps)
VAL_SPLIT  = 0.1   # fraction of replays held out for validation
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------

# Controller inputs for the "self" player — becomes the "action" HDF5 key.
# Excludes d-pad and START because they are never used in competitive play.
ACTION_COLS = [
    "self_main_x", "self_main_y",    # main analog stick
    "self_c_x",    "self_c_y",       # C-stick
    "self_l_shldr", "self_r_shldr",  # analog shoulder triggers
    "self_btn_BUTTON_A",
    "self_btn_BUTTON_B",
    "self_btn_BUTTON_X",
    "self_btn_BUTTON_Y",
    "self_btn_BUTTON_Z",
    "self_btn_BUTTON_L",
    "self_btn_BUTTON_R",
]  # action_dim = 13

# Columns that must NOT appear in the observation vector.
# "frame" is the alignment key used for joining with rendered frames.
# port columns are replay-specific identifiers, not game state.
_NON_OBS = set(ACTION_COLS) | {"frame", "self_port", "opp_port"}

def get_obs_cols(all_columns: list[str]) -> list[str]:
    """Return observation columns from a parquet column list.

    Everything that is not an action column, the frame index, or a port
    identifier is considered an observation feature.
    """
    return [c for c in all_columns if c not in _NON_OBS]
