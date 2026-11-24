# Multiscale AEM Data

Research code, datasets, trained models, and figures for multiscale polymer and anion-exchange membrane (AEM) modeling.

## Contents

- **`DFT_and_COSMORS/`** — DFT and COSMO-RS inputs, outputs, and analysis for polymer–polymer and polymer–solvent interactions.
- **`SCFT/`** — Self-consistent field theory simulations, triblock datasets, Gaussian-process models, and phase-diagram visualization.
- **`Homogenization/`** — Conductivity and elasticity calculations, topology optimization, and grain-scale extrapolation.
- **`figures/`** — Research figures in PNG and SVG formats.

## Download and use

Install [Git LFS](https://git-lfs.com/) before cloning to retrieve the large model and data files:

```sh
git lfs install
git clone https://github.com/jt577/multiscale-AEM-data.git
cd multiscale-AEM-data
git lfs pull
```

Scripts and notebooks use workflow-specific dependencies and input files. Review the relevant scripts and `infile.txt` files, and update local paths before running calculations.

DFT binary files (`*.gbw` and `*.densities`), local Python environments, and caches are excluded from this repository. Trained models are included; files larger than 10 MiB are stored with Git LFS.
