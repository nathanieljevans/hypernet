# HyperNet

[![DOI](https://zenodo.org/badge/862143613.svg)](https://doi.org/10.5281/zenodo.18853388)  

A PyTorch library for hypernetwork-based uncertainty estimation in neural networks. A hypernetwork $f_\phi(z)$ maps stochastic latent samples $z \sim p(z)$ to weight vectors $\theta$ for a base network, producing an implicit ensemble of models that yields calibrated uncertainty estimates through prediction variance.

## Key Features

- **Vectorized hypernetwork** using `torch.func.vmap` and `functional_call` for fast parallel sampling
- **Learnable priors** via Real NVP normalizing flows over the latent space $p(z)$
- **Multiple loss objectives** — MSE, L1, Gaussian NLL, Energy Distance, and Cross-Entropy
- **Baseline methods** — MC Dropout (MCDO) and MC BatchNorm (MCBN) for comparison
- **Synthetic data generators** and plotting utilities for 1D regression experiments

## Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate hnet

# Install the package in editable mode
pip install -e .
```

## Quick Start

See `notebooks` for examples.

## Architecture

```
z ~ p(z)  ──►  [Optional: RealNVP flow]  ──►  f_φ(z)  ──►  θ  ──►  base_model(x; θ)  ──►  ŷ
                                                 │
                                          3-layer MLP
                                       (z_dim → width → width → n_params)
```

The hypernetwork `f_φ` is a small MLP that maps latent codes to weight vectors for the base model. At inference, multiple latent samples produce multiple weight sets, and predictions are aggregated for mean estimates and uncertainty quantification.

**HyperNet parameters:**

| Parameter | Description | Default |
|---|---|---|
| `stochastic_channels` | Dimension of $z$ | `8` |
| `width` | Hidden size of $f_\phi$ | `10` | (Single layer MLP)
| `pz` | Prior type: `normal`, `uniform`, `bernoulli`, `categorical` | `normal` |
| `learn_pz` | Learn the prior via normalizing flows (RealNVP) | `False` |
| `affine` | Learnable scale on output weights | `False` |
| `norm` | Normalization in $f_\phi$: `none` or `layer` | `none` |
| `dropout` | Dropout rate in $f_\phi$ | `0` |

**Useful loss functions** 

We find that energy distance loss is a very useful loss function for use with Hypernetworks as it prevents mode collapse (compared to MSE) and can effectively learn both epistemic and aleatoric uncertainty. Example usage: 

```python 
from hnet.train.hnet import EnergyDistanceLoss 
import torch

x = torch.randn(100, 256, 2)    # (n_samples, batch_size, 2)
y = torch.randn(256, 2)         # (batch_size, 2)
edl = EnergyDistanceLoss()

loss = edl(x,y)
loss # scalar tensor
```

**Other notes** 

The `vmap` function does not play well with batchnorm layers, and will fail if any base model uses them. We suggest avoiding batchnorm layers for now and will explore work arounds in the future (especially if users request this feature). 

## Project Structure

```
hnet/
├── models/
│   ├── HyperNet.py        # Vectorized hypernetwork (vmap)
│   ├── HyperNet_.py       # Loop-based variant (BatchNorm compatible)
│   ├── MLP.py             # Base MLP with configurable depth/norm/activation
│   ├── RealNVP.py         # Real NVP normalizing flow for learned priors
│   ├── BSpline.py         # B-spline basis layer
│   ├── MCDO.py            # MC Dropout baseline
│   └── MCBN.py            # MC BatchNorm baseline
├── train/
│   ├── hnet.py            # HyperNet training loop + EnergyDistanceLoss
│   ├── mcdo.py            # MC Dropout training
│   ├── mcbn.py            # MC BatchNorm training
│   ├── run.py             # Tabular dataset runner (diabetes, california, mnist)
│   └── utils.py           # Expected calibration error
└── synth/
    ├── generate.py        # 1D synthetic regression data
    └── utils.py           # Plotting utilities
```

## License

MIT License — see [LICENSE](LICENSE) for details.
