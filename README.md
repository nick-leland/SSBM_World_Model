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

# Resources
Libmelee: https://github.com/vladfi1/libmelee/
LeWorldModel Paper: https://arxiv.org/pdf/2603.19312v1
Eric Gu's Melee Bot Writeup: https://ericyuegu.com/melee-pt1
Dataset of Tournament Matches: https://huggingface.co/datasets/erickfm/slippi-public-dataset-v3.7
