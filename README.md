# HSM Prototype

## Install
```
pip install torch torchvision datasets pyyaml tqdm
```

## Quick start (synthetic, no data download needed)
```
python train.py --config configs/hsm_synthetic_tiny.yaml --mode full
```

## LRA ListOps
```
python train.py --config configs/hsm_lra_listops_tiny.yaml --mode full
```

## Run all ablations
```
for mode in full local_only global_only fixed_mix no_reg; do
  python train.py --config configs/hsm_synthetic_tiny.yaml --mode $mode
done
```

## Evaluate
```
python eval.py --results_dir runs/
```

## Architecture

**HSM Layer**: Each layer applies pre-LayerNorm, then routes through:
- **ISG** (Intrinsic State Gate): computes 4 token-geometry features (L2², sequence variance, temporal diff², channel variance), normalizes via running stats, and produces a 3-way softmax gate `[g_l, g_g, g_s]`.
- **Local Mixer**: Multi-scale depthwise separable convolutions (kernels 3, 7, 15) along the sequence dimension.
- **Global Mixer**: Diagonal SSM with learnable state-transition, input/output projections, and a skip connection.
- **Merge**: `y = g_l * L(z) + g_g * G(z)` (g_s acts as implicit skip/bypass).

**Homeostatic Loss**: Task loss + regularization on gate distribution to maintain target utilization.

## Ablation Modes
| Mode         | Description                                     |
|--------------|-------------------------------------------------|
| `full`       | Full ISG-gated model                            |
| `local_only` | Always g_l=1, g_g=0 (pure local conv)          |
| `global_only`| Always g_l=0, g_g=1 (pure SSM)                 |
| `fixed_mix`  | g_l=g_g=0.5, no ISG                            |
| `no_reg`     | Full ISG but reg loss weights set to 0          |
