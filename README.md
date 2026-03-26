# SSBM_World_Model
A LeWM-style world model for Super Smash Bros. Melee.

## Architecture

The model takes both a rendered game frame and structured game state features as encoder input, predicts the next game state in latent space, and decodes predictions back to structured state values. Dolphin acts as a free, lossless renderer — since game state → frame is deterministic, we never need to learn a pixel decoder.

```
[frame_t + state_t] → Encoder → z_t
                                  |
               Predictor(z_t, a_t) → z_hat_{t+1}
                                          |
                        Decoder → predicted state_{t+1}

Loss: MSE(predicted_state, actual_state) + KL(z || N(0, I))
Viz:  Dolphin(predicted_state) → predicted frame (no learning needed)
```

**Why this works well for Melee:**
- LeWM's JEPA approach avoids learning a pixel decoder because pixel reconstruction is noisy and destabilizes training. In Melee, Dolphin *is* the decoder — game state → frame is exact and free, so we supervise predictions against structured state values (positions, velocities, actions) instead.
- Visual frames still enrich the encoder: animations, hitbox timing, and stage state are easier to read from pixels than from raw action enums.
- The prediction target (structured game state) is low-dimensional and exactly supervised, giving clean gradients.

**Training objective (LeWM):**
- `L_pred`: MSE between predicted and actual next game state
- `L_reg`: KL divergence from N(0, I) on the latent embeddings (prevents collapse without a momentum encoder)

## Preprocessing Pipeline

Converts raw Slippi `.slp` replay files into HDF5 datasets for le-wm training.

```
.slp files
    ├── extract_states.py  →  data/states/*.parquet   (game state, ~250 cols/frame)
    └── render_frames.py   →  data/frames/{id}/*.png  (rendered via Dolphin)
                  ↓
            build_hdf5.py  →  data/hdf5/melee_{train,val}.h5
```

**Quick start (training machine):**
```bash
# Download ~1k replays and extract game state (no Dolphin needed yet):
python preprocess/download.py --n-replays 1000
python preprocess/extract_states.py
python preprocess/build_hdf5.py   # uses black placeholder frames if Dolphin not set up

# Full pipeline once Dolphin + SSBM ISO are available:
DOLPHIN_BIN=/path/to/dolphin MELEE_ISO=/path/to/melee.iso bash preprocess/run_all.sh
```

**HDF5 schema** (`data/hdf5/melee_train.h5`):

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `pixels` | `(T, 224, 224, 3)` | uint8 | Raw rendered frames (ImageNet norm applied at load time) |
| `action` | `(T, 13)` | float32 | Z-score normalised controller inputs (stick, triggers, buttons) |
| `observation` | `(T, obs_dim)` | float32 | Z-score normalised game state (positions, velocities, stage, projectiles) |
| `offsets` | `(N+1,)` | int64 | Episode boundary indices into the flat T dimension |

Normalisation statistics are saved to `data/hdf5/norm_stats.json`.
Frame rendering requires [Slippi Playback Dolphin](https://slippi.gg) and an SSBM NTSC 1.02 ISO.

## Repository Structure

```
preprocess/         preprocessing pipeline scripts
le-wm/              cloned LeWorldModel repo (with melee.yaml config added)
slippi-frame-extractor/  game state extraction from .slp files
data/               generated data (gitignored)
  raw/              downloaded .slp files + manifest.json
  states/           per-frame parquet files
  frames/           rendered PNGs (one subdir per replay)
  hdf5/             final HDF5 datasets + norm_stats.json
```

## Resources
- Libmelee: https://github.com/vladfi1/libmelee/
- LeWorldModel Paper: https://arxiv.org/pdf/2603.19312v1
- Eric Gu's Melee Bot Writeup: https://ericyuegu.com/melee-pt1
- Dataset of Tournament Matches: https://huggingface.co/datasets/erickfm/slippi-public-dataset-v3.7
