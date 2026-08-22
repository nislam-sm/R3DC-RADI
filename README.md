<div align="center">

# R³DC
### Knowing What to Revise-Reliability-Aware Depth Completion for Trustworthy Cross-Domain Sparse Perception

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.3%2B-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-arXiv-orange.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![Venue](https://img.shields.io/badge/CVPR%202026-3D%20Geometry%20Generation%20Workshop-purple.svg)](#-citation)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**[Noor Islam S. Mohammad](mailto:mohammadn@itu.edu.tr)** · **[Uluğ Bayazıt](mailto:bayazit@itu.edu.tr)**

Department of Computer Engineering, Istanbul Technical University (İTÜ)

[Overview](#-overview) · [Install](#-installation) · [Datasets](#-dataset-preparation) · [Train](#-training) · [Evaluate](#-evaluation) · [RADI](#-radi-protocol) · [Results](#-results) · [FAQ](#-faq)

</div>

---

> **TL;DR** — Most depth-completion networks emit a confidence map as a *terminal* by-product that nothing downstream ever reads. R³DC turns reliability into a **control signal**: the network first *reveals* where it is likely to be wrong, then *revises* exactly those places through a reliability-gated CSPN++ propagation stage. Reliability is learned **without reliability labels**, purely from the downstream refinement objective. We also release **RADI**, an evaluation protocol that asks not only *"is the confidence calibrated?"* but *"did the confidence actually change the output?"*

---

## 📋 Table of Contents

<!-- toc -->

1. [Overview](#-overview)
   - [Motivation](#motivation)
   - [Key Innovations](#key-innovations)
   - [Comparison with Prior Work](#comparison-with-prior-work)
2. [Key Features](#-key-features)
3. [Method](#-method)
   - [Problem Formulation](#problem-formulation)
   - [Reveal Stage](#stage-1--reveal)
   - [Revise Stage](#stage-2--revise)
   - [Why Reliability Emerges Without Labels](#why-reliability-emerges-without-labels)
   - [Loss Functions](#loss-functions)
4. [Architecture](#-architecture)
   - [Block Diagram](#block-diagram)
   - [Component Details](#component-details)
   - [Tensor Shape Reference](#tensor-shape-reference)
   - [Model Variants](#model-variants)
5. [Repository Structure](#-repository-structure)
6. [Installation](#-installation)
7. [Dataset Preparation](#-dataset-preparation)
8. [Quick Start](#-quick-start)
9. [Training](#-training)
10. [Evaluation](#-evaluation)
11. [Inference](#-inference)
12. [Configuration Reference](#-configuration-reference)
13. [Results](#-results)
14. [Ablation Studies](#-ablation-studies)
15. [RADI Protocol](#-radi-protocol)
16. [Reproducibility](#-reproducibility)
17. [Troubleshooting](#-troubleshooting)
18. [FAQ](#-faq)
19. [Roadmap](#-roadmap)
20. [Contributing](#-contributing)
21. [Citation](#-citation)
22. [License](#-license)
23. [Acknowledgments](#-acknowledgments)
24. [Contact](#-contact)

<!-- tocstop -->

---

## 🎯 Overview

**R³DC** (*Reliability-Aware **R**eveal-to-**R**evise **D**epth **C**ompletion*) is a depth-completion framework that jointly predicts (i) a dense metric depth map, (ii) a per-pixel **reliability** map, and (iii) a per-pixel **aleatoric uncertainty** map — and then *feeds the reliability map back into the network* to steer a spatial-propagation refinement stage.

### Motivation

Sparse-to-dense depth completion is deployed in exactly the settings where silent failure is most expensive: autonomous driving, UAV navigation, robotic manipulation. Yet the dominant design pattern treats trustworthiness as an afterthought:

| Common pattern | Consequence |
|---|---|
| Confidence predicted as a side output, never consumed | Confidence can be arbitrarily wrong without hurting the training loss |
| Uncertainty via MC-Dropout / Deep Ensembles | 3×–20× inference cost, still no effect on the depth itself |
| Uniform refinement over all pixels | Compute is spent on already-correct regions; hard regions get the same treatment as easy ones |
| Per-dataset recipes and per-dataset architectures | "Generalization" claims that do not survive a domain change |

R³DC is built on a single hypothesis:

> **If a reliability map is forced to *do work*, it must become meaningful.**

By making reliability the gate on the refinement operator, the network can only reduce its own depth loss by placing low reliability where it is genuinely wrong. No reliability supervision is needed — the signal is created by the architecture.

### Key Innovations

1. **Reliability as a control signal.** Predicted reliability `R̂ ∈ (0,1)` modulates the affinity weights of a CSPN++ propagation network, so low-reliability pixels absorb more information from their high-reliability neighbours.
2. **Closed-loop, label-free reliability learning.** No ground-truth confidence maps, no auxiliary reliability loss, no post-hoc calibration set. Reliability is learned solely through the refinement loss on `D₁`.
3. **RADI evaluation protocol.** Three complementary axes — **REC** (does the model know where it is wrong?), **RBS** (does it act on that knowledge?), **CAL** (are the numbers meaningful?) — computed globally and over four difficulty-stratified regions.
4. **Cross-domain generalization from one recipe.** The same architecture, loss and schedule train on automotive LiDAR (KITTI), indoor structured light (NYU-v2), and aerial imagery with synthesized sparse priors (VisDrone, Drone-Videos).
5. **Parameter efficiency.** 1.95 M parameters for the base model; 11.22 M for the extended model that matches or beats far larger baselines.
6. **Indoor Calibration Head (ICH).** A 16,642-parameter adapter that lifts a frozen monocular foundation backbone (Depth-Anything-V2) to metric, sparse-conditioned indoor depth.

### Comparison with Prior Work

| Capability | **R³DC** | CSPN++ | NLSPN | PENet | CompletionFormer |
|---|:--:|:--:|:--:|:--:|:--:|
| Per-pixel reliability output | ✅ | ❌ | ❌ | ❌ | ❌ |
| Reliability **consumed** by refinement | ✅ | ❌ | ❌ | ❌ | ❌ |
| Single-pass uncertainty (no MC / ensembling) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Aleatoric uncertainty head | ✅ | ❌ | ❌ | ❌ | ❌ |
| One recipe across 4 domains | ✅ | ❌ | ❌ | ❌ | ❌ |
| Sub-2M-parameter operating point | ✅ | ❌ | ❌ | ❌ | ❌ |
| Reliability-aware evaluation protocol | ✅ | ❌ | ❌ | ❌ | ❌ |

> **Honesty note.** R³DC-base (1.95 M) does **not** beat the strongest baselines on raw KITTI RMSE — it trades ~8 % accuracy for a 6–67× parameter reduction plus reliability. R³DC+ (11.22 M) is competitive with CompletionFormer while adding reliability at no inference-time cost. We report both, and we consider the honest reporting of the small model's deficit part of the contribution.

---

## 🚀 Key Features

- 🎯 **Reliability-guided refinement** — CSPN++ affinities gated by learned per-pixel reliability.
- 🔁 **Reveal-to-Revise two-stage decoding** — coarse depth `D₀` → gated propagation → refined depth `D₁`.
- ⚡ **Parameter-efficient** — 1.95 M (base) / 11.22 M (extended); single forward pass for depth + reliability + uncertainty.
- 🌍 **Four domains, one recipe** — KITTI, NYU Depth V2, VisDrone, Drone-Videos.
- 📊 **RADI protocol** — REC / RBS / CAL / AUSE over `all`, `edge`, `textureless`, `far-depth` regions.
- 🏠 **Indoor Calibration Head** — 16,642-parameter foundation-model adapter.
- 🔬 **Physics-motivated synthetic depth** — generates plausible sparse priors for RGB-only aerial datasets.
- 🔄 **EMA inference** — exponential moving average weights (`decay = 0.9999`) for stable evaluation.
- 🧰 **Production-grade tooling** — AMP, DDP, gradient clipping, cosine warm restarts, W&B / TensorBoard logging, deterministic seeding, resumable checkpoints.
- 🧪 **Full CLI** — `train.py`, `evaluate.py`, `inference.py`, `inference_batch.py`, plus dataset preparation scripts.

---

## 🧠 Method

### Problem Formulation

Given an RGB image `I ∈ ℝ^{3×H×W}` and a sparse depth map `S ∈ ℝ^{1×H×W}` with validity mask `M ∈ {0,1}^{1×H×W}` (typically 3–8 % valid pixels for LiDAR), predict a dense depth map `D ∈ ℝ^{1×H×W}` that agrees with the (semi-dense) ground truth `D*` on its valid support.

R³DC additionally predicts:

- **Reliability** `R̂ ∈ (0,1)^{1×H×W}` — a *relative* trust score used internally as a gate.
- **Aleatoric uncertainty** `σ̂ ∈ ℝ₊^{1×H×W}` — the observation-noise scale of a heteroscedastic Laplacian likelihood.

Depth is regressed in a **normalized log space** for numerical stability:

```
d̃ = ( log(d + ε) − log(d_min + ε) ) / ( log(d_max + ε) − log(d_min + ε) ) ∈ [0, 1]
```

with the inverse mapping applied before any metric evaluation. `d_min`, `d_max` and `ε` are dataset-specific (see [Configuration Reference](#-configuration-reference)).

### Stage 1 — Reveal

A dual-branch encoder ingests RGB and (sparse depth ⊕ mask) separately, fuses them with **Cross-Modal Attention (CMA)** at three scales, passes through a **transformer bottleneck**, and decodes through an **FPN decoder**. Four heads read the decoder output:

| Head | Activation | Output | Purpose |
|---|---|---|---|
| Depth | Sigmoid | `D₀ ∈ [0,1]` | Coarse normalized depth |
| Reliability | Sigmoid | `R̂ ∈ (0,1)` | Where the model expects to be right |
| Uncertainty | Softplus | `σ̂ > 0` | Aleatoric noise scale |
| Auxiliary (×2) | Sigmoid | `D₀^{1/2}`, `D₀^{1/4}` | Deep supervision |

### Stage 2 — Revise

An affinity network predicts a `k×k` neighbourhood affinity tensor `A` (here `k = 3`, so 8 neighbours). The **reliability gate** rescales each neighbour's influence by the *relative* trust of source vs. target:

```
Ã_{p→q}  =  A_{p→q} · R̂_q / ( R̂_p + R̂_q + ε )
```

so information flows **from confident pixels into unconfident ones**, not the reverse. Affinities are then normalized so the propagation is stable:

```
Σ_{q ∈ N(p)} |Ã_{p→q}|  ≤  1 − w_c ,        w_c = 0.2  (fixed center weight)
```

The propagation runs for `T = 6` steps:

```
D^{(t+1)}_p  =  w_c · D^{(t)}_p  +  Σ_{q ∈ N(p)} Ã_{p→q} · D^{(t)}_q
```

with **Dirichlet boundary conditions** — measured sparse pixels are re-clamped to their observed value after every step:

```
D^{(t+1)}  ←  M ⊙ S̃  +  (1 − M) ⊙ D^{(t+1)}
```

The output of the final step is the refined depth `D₁`.

### Why Reliability Emerges Without Labels

`R̂` appears **only** inside the gate. The gradient of the refinement loss w.r.t. `R̂` is therefore non-zero only through its effect on the propagation:

- Marking a pixel **low-reliability** where `D₀` is already accurate → it gets overwritten by neighbours → loss ↑
- Marking a pixel **high-reliability** where `D₀` is wrong → the error is broadcast to neighbours → loss ↑
- The only loss-minimizing configuration is one where `R̂` anti-correlates with the true error of `D₀`.

This is what RADI's **RBS** metric measures directly, and it is why hand-crafted or post-hoc confidence proxies score `RBS = 0 %` in our comparisons: they were never in the loop.

### Loss Functions

The composite objective:

```
L = λ_silog·L_SILog + λ_fb·L_FocalBerHu + λ_ssim·L_SSIM + λ_anchor·L_anchor
  + λ_vnl·L_VNL + λ_dnc·L_DNC + λ_grad·L_grad + λ_unc·L_unc + λ_aux·L_aux
```

applied to **both** `D₀` and `D₁` (the `D₁` term carries the reliability gradient).

| Term | Formula / description | Default λ |
|---|---|---|
| **SILog** | `√( mean(g²) − 0.85·mean(g)² ) · 10`, `g = log d̂ − log d*` | 1.00 |
| **Focal-BerHu** | BerHu (reverse Huber) with threshold `c = 0.2·max|e|`, re-weighted by a focal factor `(e/max e)^γ`, `γ = 2` | 0.60 |
| **SSIM** | `(1 − SSIM(d̂, d*)) / 2`, 7×7 window — structural fidelity | 0.20 |
| **Anchor** | L1 on measured sparse pixels — prevents drift away from LiDAR returns | 0.15 |
| **VNL** | Virtual Normal Loss over 1024 random point triplets — global geometric consistency | 0.10 |
| **DNC** | Depth-Normal Consistency between predicted depth gradients and surface normals | 0.05 |
| **Grad** | Multi-scale gradient matching — sharpens discontinuities | 0.05 |
| **Unc** | Heteroscedastic Laplacian NLL: `|e|/σ̂ + log σ̂` | 0.05 |
| **Aux** | Deep supervision at 1/2 and 1/4 resolution | 0.10 |

> ⚠️ `L_unc` trains `σ̂` only; it is **detached** from the depth path so uncertainty cannot down-weight the depth loss into a degenerate solution.

---

## 🏗️ Architecture

### Block Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              R³DC ARCHITECTURE                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   RGB (3×H×W) ─────────► RGB Encoder ────────┐                               │
│                          (3 stages, residual │                               │
│                           + DropPath)        │                               │
│                                              ├─► CMA ×3 ─► Transformer ─┐    │
│                                              │   (tokens   Bottleneck   │    │
│   Sparse Depth ⊕ Mask ─► Depth Encoder ──────┘    ≤512)    (8 heads,    │    │
│   (2×H×W)                (DCNv2, sparse-aware)              CBAM)       │    │
│                                                                          │    │
│                                    ┌─────────────────────────────────────┘    │
│                                    ▼                                          │
│                            ┌───────────────┐                                  │
│                            │  FPN Decoder  │  4 × EfficientUpBlock            │
│                            │  1/8→1/4→1/2→1│  (Up + DCN fuse + Res            │
│                            └───────┬───────┘   + CBAM + CMA)                  │
│                                    │                                          │
│         ┌──────────────┬───────────┼───────────┬──────────────┐              │
│         ▼              ▼           ▼           ▼              ▼              │
│   ┌──────────┐  ┌─────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│   │  Depth   │  │ Reliability │ │Uncertain.│ │  Aux 1/2 │ │  Aux 1/4 │      │
│   │   Head   │  │    Head     │ │   Head   │ │   Head   │ │   Head   │      │
│   │ →  D₀    │  │ →   R̂       │ │ →  σ̂     │ │          │ │          │      │
│   └────┬─────┘  └──────┬──────┘ └────┬─────┘ └──────────┘ └──────────┘      │
│        │               │             │        (deep supervision only)        │
│        │        ┌──────┴──────┐      │                                       │
│        │        ▼             │      │                                       │
│        │  ┌───────────────────▼──┐   │                                       │
│        └─►│   Affinity Network    │  │                                       │
│           │   A  ──gate by R̂──► Ã  │  │                                       │
│           └───────────┬───────────┘  │                                       │
│                       ▼              │                                       │
│           ┌───────────────────────┐  │                                       │
│           │ Reliability-Gated     │  │                                       │
│           │ CSPN++  (T = 6 steps, │  │                                       │
│           │ Dirichlet anchors,    │  │                                       │
│           │ w_c = 0.2)            │  │                                       │
│           └───────────┬───────────┘  │                                       │
│                       ▼              ▼                                       │
│              Refined Depth D₁    Uncertainty σ̂                               │
│                       │                                                      │
│                       ▼                                                      │
│           Inverse log-normalization  ──►  Metric depth (m)                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Component Details

<details>
<summary><b>Encoder</b> (click to expand)</summary>

| Module | Configuration |
|---|---|
| **RGB Encoder** | 3-stage residual network, channels `[W, 2W, 4W]` with `W = base_width`; GroupNorm(32); stochastic depth `drop_path_rate = 0.1` |
| **Depth Encoder** | Mirrors the RGB stages but uses **Deformable Convolution v2 (DCNv2)** so receptive fields adapt to the irregular sparse support; the validity mask is concatenated as a second input channel |
| **Cross-Modal Attention (CMA)** | Applied at 3 scales; RGB features form queries, depth features form keys/values; token budget capped at `max_tokens = 512` via adaptive pooling to bound cost at high resolution |
| **Transformer Bottleneck** | Multi-head self-attention, `num_heads = 8`, pre-norm, followed by **CBAM** (channel + spatial attention) |

</details>

<details>
<summary><b>Decoder</b> (click to expand)</summary>

| Module | Configuration |
|---|---|
| **FPN Decoder** | 4 × `EfficientUpBlock`, lateral skip connections from both encoders |
| **EfficientUpBlock** | `ConvTranspose2d` upsample → DCN-based skip fusion → residual block → CBAM → CMA |
| **Multi-scale outputs** | Full, 1/2, 1/4, 1/8 resolution feature maps; the 1/2 and 1/4 maps feed the auxiliary heads |

</details>

<details>
<summary><b>Output heads</b> (click to expand)</summary>

| Head | Structure | Activation |
|---|---|---|
| Depth | `3×3 Conv → ReLU → 1×1 Conv` | Sigmoid |
| Reliability | `3×3 Conv → ReLU → 1×1 Conv` | Sigmoid |
| Uncertainty | `3×3 Conv → ReLU → 1×1 Conv` | Softplus |
| Auxiliary ×2 | `3×3 Conv → ReLU → 1×1 Conv` | Sigmoid |

</details>

<details>
<summary><b>Refinement</b> (click to expand)</summary>

| Property | Value |
|---|---|
| Propagation steps `T` | 6 |
| Kernel size `k` | 3 (8-neighbourhood) |
| Center weight `w_c` | 0.2 (fixed, not learned — critical for stability) |
| Affinity normalization | `Σ|Ã| ≤ 1 − w_c` |
| Boundary condition | Dirichlet — sparse anchors re-clamped every step |
| Gate | `Ã_{p→q} = A_{p→q} · R̂_q / (R̂_p + R̂_q + ε)` |

</details>

<details>
<summary><b>Indoor Calibration Head (ICH)</b> (click to expand)</summary>

For NYU Depth V2 we optionally freeze a **Depth-Anything-V2** backbone and attach a 16,642-parameter adapter that:

1. Takes the relative-depth prediction of the foundation model,
2. Conditions on the sparse metric anchors,
3. Predicts a per-image scale/shift plus a low-rank spatial residual,
4. Emits metric depth **and** the reliability map that drives the same gated CSPN++ stage.

This is how R³DC reaches `δ₁ = 0.927` on NYU while training only ~0.02 % of the total parameters.

</details>

### Tensor Shape Reference

| Symbol | Shape | Range | Notes |
|---|---|---|---|
| `rgb` | `(B, 3, H, W)` | `[0,1]` after ImageNet norm | float32 |
| `sparse_depth` | `(B, 1, H, W)` | metres, 0 = missing | raw sensor units |
| `sparse_mask` | `(B, 1, H, W)` | `{0,1}` | `sparse_depth > 0` |
| `d0` | `(B, 1, H, W)` | `[0,1]` | normalized coarse |
| `d1` | `(B, 1, H, W)` | `[0,1]` | normalized refined |
| `d0_metric`, `d1_metric` | `(B, 1, H, W)` | `[d_min, d_max]` m | inverse-normalized |
| `reliability` | `(B, 1, H, W)` | `(0,1)` | gate signal |
| `uncertainty` | `(B, 1, H, W)` | `(0, ∞)` | metres |
| `aux_half`, `aux_quarter` | `(B,1,H/2,W/2)`, `(B,1,H/4,W/4)` | `[0,1]` | training only |

### Model Variants

| Variant | Params | `base_width` | Backbone | Intended use |
|---|---:|:--:|---|---|
| `r3dc-tiny` | 0.61 M | 16 | — | Embedded / MCU-class UAV payloads |
| `r3dc-base` | **1.95 M** | 32 | — | Real-time edge deployment |
| `r3dc-plus` (**R³DC+**) | **11.22 M** | 64 | — | Benchmark-competitive setting |
| `r3dc-ich` | 94.6 M (frozen) + 16.6 K (trained) | — | Depth-Anything-V2 | Indoor / NYU foundation-model setting |

---

## 📁 Repository Structure

```
r3dc/
├── configs/                        # YAML experiment configurations
│   ├── kitti.yaml
│   ├── nyu.yaml
│   ├── nyu_ich.yaml                # Depth-Anything-V2 + Indoor Calibration Head
│   ├── visdrone.yaml
│   ├── drone_videos.yaml
│   └── ablations/                  # One YAML per ablation row
│       ├── no_gate.yaml
│       ├── no_dirichlet.yaml
│       ├── steps_{2,4,8,12}.yaml
│       └── ...
├── data/                           # Datasets (git-ignored)
├── datasets/
│   ├── __init__.py                 # build_dataset() registry
│   ├── base.py                     # Shared augmentation + normalization
│   ├── kitti.py
│   ├── nyu.py
│   ├── visdrone.py
│   ├── drone_videos.py
│   └── synthetic_depth.py          # Physics-motivated sparse-prior generator
├── models/
│   ├── r3dc.py                     # Top-level R3DC module
│   ├── encoders.py                 # RGB / sparse-depth encoders
│   ├── attention.py                # CMA, CBAM, transformer bottleneck
│   ├── decoder.py                  # FPN decoder, EfficientUpBlock
│   ├── heads.py                    # Depth / reliability / uncertainty / aux
│   ├── cspn.py                     # Reliability-gated CSPN++
│   ├── ich.py                      # Indoor Calibration Head
│   └── layers/
│       ├── dcn.py                  # DCNv2 wrapper
│       └── droppath.py
├── losses/
│   ├── composite.py                # Weighted sum + scheduling
│   ├── silog.py
│   ├── focal_berhu.py
│   ├── ssim.py
│   ├── vnl.py
│   ├── dnc.py
│   └── uncertainty.py
├── metrics/
│   ├── depth.py                    # RMSE / MAE / AbsRel / SILog / δₙ / iRMSE / iMAE
│   └── radi.py                     # RADI: REC, RBS, CAL, AUSE + region masks
├── utils/
│   ├── config.py                   # load_config, dict_to_namespace
│   ├── checkpoint.py               # save/load, EMA handling
│   ├── ema.py
│   ├── logger.py                   # console + TensorBoard + W&B
│   ├── distributed.py              # DDP helpers
│   ├── seed.py
│   └── visualization.py            # Colormaps, panel figures
├── scripts/
│   ├── prepare_kitti_splits.py
│   ├── download_nyu.py
│   ├── prepare_visdrone.py
│   ├── prepare_drone_videos.py
│   ├── export_onnx.py
│   ├── benchmark_latency.py
│   └── make_figures.py
├── tests/
│   ├── test_shapes.py
│   ├── test_cspn_stability.py
│   ├── test_radi.py
│   └── test_normalization.py
├── train.py
├── evaluate.py
├── inference.py
├── inference_batch.py
├── requirements.txt
├── setup.py
├── LICENSE
├── CONTRIBUTING.md
└── README.md
```

---

## 📦 Installation

### Prerequisites

```bash
python --version     # ≥ 3.8
nvidia-smi           # CUDA ≥ 11.3 recommended for training
gcc --version        # ≥ 7 (needed to build DCNv2 if no wheel is available)
```

| Resource | Minimum | Recommended |
|---|---|---|
| GPU VRAM (training, KITTI, `bs=4`, AMP) | 11 GB | 24 GB |
| GPU VRAM (inference, base model) | 2 GB | 4 GB |
| System RAM | 16 GB | 32 GB |
| Disk (all four datasets) | ~250 GB | 500 GB SSD |

### Option A — pip + venv

```bash
git clone https://github.com/yourusername/r3dc.git
cd r3dc

python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .                    # editable install exposes the `r3dc` package
```

### Option B — conda

```bash
conda create -n r3dc python=3.10 -y
conda activate r3dc

conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
pip install -e .
```

### Option C — Docker

```bash
docker build -t r3dc:latest .

docker run --gpus all -it --rm \
    --shm-size=16g \
    -v $(pwd):/workspace \
    -v /path/to/datasets:/workspace/data \
    r3dc:latest bash
```

<details>
<summary><b>Reference <code>Dockerfile</code></b></summary>

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git ninja-build libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install -e .

CMD ["bash"]
```

</details>

### Requirements

```txt
# ---- Core ----
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.19.0,<2.0

# ---- Image processing ----
opencv-python>=4.5.0
Pillow>=8.0.0
scipy>=1.7.0
scikit-image>=0.19.0

# ---- Metrics / stats ----
scikit-learn>=1.0.0

# ---- Visualization ----
matplotlib>=3.3.0
tqdm>=4.60.0

# ---- Configuration ----
pyyaml>=5.0
easydict>=1.9

# ---- Optional ----
wandb                # experiment tracking
tensorboard          # local logging
h5py                 # NYU .mat/.h5 loading
timm>=0.9.0          # foundation backbones (ICH path)
onnx, onnxruntime    # export + deployment
```

### Verify the Installation

```bash
# Smoke test: builds the model, runs a random forward pass, prints shapes and param count
python -m tests.test_shapes

# Expected tail of output:
#   d1_metric      : torch.Size([2, 1, 352, 1216])
#   reliability    : torch.Size([2, 1, 352, 1216])
#   uncertainty    : torch.Size([2, 1, 352, 1216])
#   Trainable parameters: 1,950,xxx
```

```bash
# Full unit-test suite
pytest tests/ -v
```

---

## 📊 Dataset Preparation

All datasets resolve under `dataset.data_root` (default `./data`). Every loader expects a `splits/` folder with newline-separated relative file ids.

### 1. KITTI Depth Completion

```bash
mkdir -p data/kitti && cd data/kitti

# Download from the official KITTI portal (registration required):
#   data_depth_velodyne.zip     (sparse LiDAR projections)
#   data_depth_annotated.zip    (semi-dense ground truth)
#   data_depth_selection.zip    (official val/test selection)
#   raw RGB sequences

unzip 'data_depth_*.zip'
cd ../..

python scripts/prepare_kitti_splits.py --root data/kitti
```

```
data/kitti/
├── image/                  # [file_id].png   RGB, uint8
├── depth/                  # [file_id].png   sparse LiDAR, uint16 (value/256 = metres)
├── gt/                     # [file_id].png   semi-dense GT, uint16
└── splits/
    ├── train.txt           # 85,898 samples
    ├── val.txt             #  1,000 samples (official selection)
    └── test.txt            #  1,000 samples (no GT — server submission)
```

> **Depth encoding.** KITTI stores depth as `uint16` PNG with `depth_m = png_value / 256.0`; `0` means *no measurement*, never *zero depth*. The loader enforces this.

### 2. NYU Depth V2

```bash
mkdir -p data/nyu && cd data/nyu
python ../../scripts/download_nyu.py     # downloads + extracts the standard split
cd ../..
```

```
data/nyu/
├── image/                  # [scene]_[frame].png
├── depth/                  # sparse sample of GT (500 points by default)
├── gt/                     # dense Kinect GT, inpainted
└── splits/
    ├── train.txt           # 47,584 samples
    └── val.txt             #    654 samples (Eigen test split)
```

> **Sparse sampling.** NYU has no native sparse sensor. Following standard practice we uniformly sample `N = 500` valid GT pixels per image at load time; `dataset.num_sparse_samples` controls `N`. Sampling is re-randomized every epoch during training and **fixed by seed** during evaluation.

### 3. VisDrone (aerial, RGB-only)

```bash
mkdir -p data/visdrone && cd data/visdrone
# Download VisDrone-DET / VisDrone-VID from the official site
cd ../..
python scripts/prepare_visdrone.py --root data/visdrone
```

```
data/visdrone/
├── images/                 # [file_id].jpg
└── splits/
    ├── train.txt
    └── val.txt
```

### 4. Drone-Videos (aerial, RGB-only)

```bash
mkdir -p data/drone_videos && cd data/drone_videos
# Download from Kaggle, then extract frames
cd ../..
python scripts/prepare_drone_videos.py --root data/drone_videos --fps 2
```

```
data/drone_videos/
├── images/                 # [file_id].jpg
└── splits/
    ├── train.txt
    └── val.txt
```

### Synthetic Depth for Aerial Domains

VisDrone and Drone-Videos ship RGB only. `datasets/synthetic_depth.py` produces a **physics-motivated** pseudo-ground-truth plus a sparse prior:

| Stage | Description |
|---|---|
| 1. Ground-plane prior | Pinhole model with assumed altitude and pitch gives a baseline depth ramp over the image |
| 2. Structure modulation | Vertical-edge density and local texture statistics displace the ramp to model buildings and vegetation |
| 3. Atmospheric cue | Haze-consistent contrast attenuation refines the far field |
| 4. Sparse sampling | Blue-noise sampling at the target density, plus range-dependent noise `σ ∝ d²` mimicking a lightweight aerial LiDAR |

```bash
# Preview the synthetic prior before training
python -m datasets.synthetic_depth \
    --image data/visdrone/images/0000001.jpg \
    --altitude 60 --pitch -35 --density 0.03 \
    --save preview.png
```

> ⚠️ **Reporting caveat.** Aerial numbers are measured against *synthetic* ground truth and are **not** comparable to KITTI/NYU numbers. They demonstrate cross-domain behaviour of the reliability mechanism, not absolute metric accuracy. We state this in the paper and repeat it here.

### Dataset Summary

| Dataset | Domain | Sparse source | Train | Val | Resolution | Depth range |
|---|---|---|---:|---:|---|---|
| KITTI DC | Automotive outdoor | 64-beam LiDAR (~5 %) | 85,898 | 1,000 | 352×1216 | 0–80 m |
| NYU Depth V2 | Indoor | 500 sampled points | 47,584 | 654 | 518×518 | 0.001–10 m |
| VisDrone | Aerial urban | Synthetic (3 %) | ~6,470 | ~548 | 384×640 | 1–80 m |
| Drone-Videos | Aerial mixed | Synthetic (3 %) | ~12,000 | ~1,500 | 384×640 | 0–50 m |

---

## ⚡ Quick Start

```bash
# 1. Install
git clone https://github.com/yourusername/r3dc.git && cd r3dc
pip install -r requirements.txt && pip install -e .

# 2. Grab a pretrained checkpoint
mkdir -p checkpoints
wget -O checkpoints/r3dc_kitti_best.pth.tar <RELEASE_URL>

# 3. Run on one image
python inference.py \
    --config configs/kitti.yaml \
    --checkpoint checkpoints/r3dc_kitti_best.pth.tar \
    --image assets/demo_rgb.png \
    --sparse assets/demo_sparse.png \
    --output_dir outputs/demo

# 4. Look at outputs/demo/demo_rgb_visualization.png
```

Five-line Python version:

```python
from r3dc import load_pretrained
model = load_pretrained("r3dc-plus-kitti")            # downloads + EMA weights + eval()
out = model.predict("assets/demo_rgb.png", "assets/demo_sparse.png")
print(out["depth"].shape, out["reliability"].mean())
```

---

## 🏋️ Training

### Single GPU

```bash
python train.py --config configs/kitti.yaml
python train.py --config configs/nyu.yaml
python train.py --config configs/visdrone.yaml
python train.py --config configs/drone_videos.yaml
```

### Multi-GPU (DDP)

```bash
# torchrun (PyTorch ≥ 1.10, preferred)
torchrun --nproc_per_node=4 train.py --config configs/kitti.yaml --ddp

# legacy launcher
python -m torch.distributed.launch --nproc_per_node=4 train.py --config configs/kitti.yaml

# pin specific devices
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py --config configs/kitti.yaml --ddp
```

> Under DDP, `dataset.batch_size` is the **per-GPU** batch size. Scale `optimizer.lr` linearly with the world size, or pass `--auto_scale_lr`.

### Resume

```bash
python train.py --config configs/kitti.yaml \
    --resume checkpoints/checkpoint_epoch_10.pth.tar
```

`--resume` restores model, EMA shadow, optimizer, scheduler, AMP scaler, epoch counter and RNG state — training continues bit-identically on the same hardware.

### Fine-tuning from another domain

```bash
python train.py --config configs/visdrone.yaml \
    --pretrained checkpoints/r3dc_kitti_best.pth.tar \
    --reset_optimizer --freeze_encoder_epochs 2
```

### Experiment Tracking

```bash
# Weights & Biases
python train.py --config configs/kitti.yaml --wandb \
    --wandb_project r3dc --wandb_name kitti_plus_baseline

# TensorBoard
tensorboard --logdir ./logs
```

### CLI Reference — `train.py`

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config` | path | *required* | YAML configuration file |
| `--resume` | path | `None` | Resume full training state from a checkpoint |
| `--pretrained` | path | `None` | Load weights only (no optimizer/epoch state) |
| `--output_dir` | path | from config | Override checkpoint + log root |
| `--ddp` | flag | `False` | Enable DistributedDataParallel |
| `--auto_scale_lr` | flag | `False` | Multiply LR by world size |
| `--amp` / `--no_amp` | flag | `True` | Toggle mixed precision |
| `--seed` | int | `42` | Global RNG seed |
| `--deterministic` | flag | `False` | cuDNN deterministic mode (slower, exactly reproducible) |
| `--wandb` | flag | `False` | Enable W&B logging |
| `--wandb_project` | str | `r3dc` | W&B project |
| `--wandb_name` | str | auto | W&B run name |
| `--freeze_encoder_epochs` | int | `0` | Warm up the decoder before unfreezing encoders |
| `--overfit_batches` | int | `0` | Debug: overfit N batches (sanity check) |
| `--dry_run` | flag | `False` | Build everything, run 2 steps, exit |

### Training Recipe Per Dataset

| Parameter | KITTI | NYU | NYU + ICH | VisDrone | Drone-Videos |
|---|---|---|---|---|---|
| Input size | 352×1216 | 518×518 | 518×518 | 384×640 | 384×640 |
| `d_min` / `d_max` | 0.0 / 80.0 m | 0.001 / 10.0 m | 0.001 / 10.0 m | 1.0 / 80.0 m | 0.0 / 50.0 m |
| Batch size | 4 | 11 | 11 | 4 | 4 |
| Epochs | 8 | 10 | 10 | 20 | 5 |
| LR | 1e-4 | 5e-6 | 5e-6 | 1e-4 | 1e-4 |
| `base_width` | 64 | — (frozen backbone) | — | 64 | 32 |
| Sparse dropout | 0.3 | 0.2 | 0.2 | 0.3 | 0.3 |
| CutMix prob | 0.3 | 0.0 | 0.0 | 0.3 | 0.3 |
| Approx. wall-clock (1×A100) | ~34 h | ~19 h | ~6 h | ~7 h | ~4 h |

### What to Watch During Training

| Metric | Healthy behaviour | Red flag |
|---|---|---|
| `loss` | Monotone decrease with warm-restart bumps | Plateau in the first 2 epochs → LR too low |
| `rmse` | Decreasing on val | Val ↑ while train ↓ → reduce augmentation strength |
| `delta_1` | → 0.85+ (KITTI) by epoch 3 | Stuck < 0.6 → check depth normalization / `d_max` |
| `reliability_mean` | Settles in `[0.45, 0.75]` | Saturated at ~1.0 → gate collapse, see [Troubleshooting](#reliability-collapse) |
| `reliability_std` | > 0.05 | ≈ 0 → the map is constant and carries no information |
| `radi_all_rec` | Positive and rising, > 0.2 by mid-training | ≈ 0 or negative → gate wired backwards |
| `radi_all_rbs` | Rising toward 30–45 % | 0 % → refinement is not using reliability |
| `grad_norm` | Stable, below `gradient_clip` most steps | Frequent clipping → lower LR |

### Training Curve Sanity Checklist

```bash
# Overfit a single batch — loss should reach ~0 within 200 steps
python train.py --config configs/kitti.yaml --overfit_batches 1

# Confirm the gate is live: reliability std must be non-zero after 500 steps
python train.py --config configs/kitti.yaml --dry_run --log_interval 1
```

---

## 📈 Evaluation

### Validation Set

```bash
# Standard depth metrics
python evaluate.py --config configs/kitti.yaml \
    --checkpoint checkpoints/model_best.pth.tar

# Depth metrics + full RADI protocol
python evaluate.py --config configs/kitti.yaml \
    --checkpoint checkpoints/model_best.pth.tar --radi

# Persist per-sample metrics + prediction dumps
python evaluate.py --config configs/kitti.yaml \
    --checkpoint checkpoints/model_best.pth.tar \
    --radi --save_results --results_dir results/kitti_val
```

### Test Set

```bash
# KITTI official benchmark (produces uint16 PNGs for server submission)
python evaluate.py --config configs/kitti.yaml \
    --checkpoint checkpoints/model_best.pth.tar \
    --split test --submission_dir submissions/kitti

# NYU Eigen test split
python evaluate.py --config configs/nyu.yaml \
    --checkpoint checkpoints/model_best.pth.tar --split test
```

### CLI Reference — `evaluate.py`

| Flag | Default | Description |
|---|---|---|
| `--config` | *required* | YAML configuration |
| `--checkpoint` | *required* | Checkpoint path (EMA weights used when present) |
| `--split` | `val` | `train` / `val` / `test` |
| `--radi` | `False` | Compute REC / RBS / CAL / AUSE and region breakdown |
| `--save_results` | `False` | Write per-sample CSV + `.npy` predictions |
| `--results_dir` | `./results` | Output root for `--save_results` |
| `--submission_dir` | `None` | Write KITTI-format uint16 PNGs |
| `--no_ema` | `False` | Evaluate raw weights instead of EMA shadow |
| `--batch_size` | from config | Override eval batch size |
| `--max_samples` | `None` | Evaluate only the first N samples (debug) |
| `--tta` | `False` | Horizontal-flip test-time augmentation |

### Example Output

```
======================================================================
                        Evaluation Results
                 config=configs/kitti.yaml  split=val
                 checkpoint=model_best.pth.tar (EMA)
                 samples=1000
======================================================================
-- Depth ------------------------------------------------------------
rmse                    : 0.240000
mae                     : 0.185000
abs_rel                 : 0.095000
silog                   : 0.080000
delta_1                 : 0.920000
delta_2                 : 0.980000
delta_3                 : 0.995000
irmse                   : 2.180000
imae                    : 0.950000
-- RADI: region breakdown -------------------------------------------
radi_all_rec            : 0.371000
radi_all_rec_p_value    : 0.000000
radi_all_rbs            : 41.300000
radi_edge_rec           : 0.358000
radi_textureless_rec    : 0.389000
radi_far_depth_rec      : 0.341000
-- RADI: global -----------------------------------------------------
radi_global_cal         : 0.041000
radi_global_ause        : 0.061000
======================================================================
```

> **Units.** `rmse` / `mae` are reported in **metres** by the CLI. The KITTI leaderboard reports **millimetres** — multiply by 1000 when comparing against the tables in [Results](#-results).

### Metric Definitions

#### Depth Metrics

Let `d̂ᵢ` be prediction and `dᵢ` ground truth over `N` valid pixels.

| Metric | Definition | Direction |
|---|---|:--:|
| **RMSE** | `√( (1/N) Σ (d̂ᵢ − dᵢ)² )` | ↓ |
| **MAE** | `(1/N) Σ \|d̂ᵢ − dᵢ\|` | ↓ |
| **AbsRel** | `(1/N) Σ \|d̂ᵢ − dᵢ\| / dᵢ` | ↓ |
| **SILog** | `√( mean(g²) − 0.85 · mean(g)² ) · 10`, `g = log d̂ − log d` | ↓ |
| **δₙ** | fraction with `max(d̂ᵢ/dᵢ, dᵢ/d̂ᵢ) < 1.25ⁿ` | ↑ |
| **iRMSE** | RMSE computed on `1/d` (1/km) | ↓ |
| **iMAE** | MAE computed on `1/d` (1/km) | ↓ |

#### RADI Metrics

| Metric | Question answered | Direction |
|---|---|:--:|
| **REC** | Does the model know *where* it is wrong? | ↑ |
| **RBS** | Does the model *act* on that knowledge? | ↑ |
| **CAL** | Are the confidence values numerically meaningful? | ↓ |
| **AUSE** | Does removing low-confidence pixels reduce error optimally? | ↓ |

Full formulas in [RADI Protocol](#-radi-protocol).

---

## 🔍 Inference

### Single Image

```bash
# RGB + sparse depth (standard completion)
python inference.py \
    --config configs/kitti.yaml \
    --checkpoint checkpoints/model_best.pth.tar \
    --image path/to/rgb.png \
    --sparse path/to/sparse_depth.png \
    --output_dir ./outputs

# RGB only (monocular fallback — the depth branch receives an all-zero mask)
python inference.py \
    --config configs/kitti.yaml \
    --checkpoint checkpoints/model_best.pth.tar \
    --image path/to/rgb.png \
    --output_dir ./outputs
```

### Batch Directory

```bash
python inference_batch.py \
    --config configs/kitti.yaml \
    --checkpoint checkpoints/model_best.pth.tar \
    --image_dir path/to/images \
    --sparse_dir path/to/sparse_depths \
    --output_dir ./outputs \
    --batch_size 8 --num_workers 8
```

### CLI Reference — `inference.py`

| Flag | Default | Description |
|---|---|---|
| `--image` | *required* | RGB image path |
| `--sparse` | `None` | Sparse depth PNG (uint16, `/256` metres); omit for monocular |
| `--output_dir` | `./outputs` | Where artefacts are written |
| `--save_npy` | `True` | Also dump raw float arrays |
| `--colormap` | `magma` | Depth colormap |
| `--reliability_cmap` | `RdYlGn` | Reliability colormap (green = trusted) |
| `--max_depth_vis` | from config | Clip value for visualization only |
| `--fp16` | `False` | Half-precision inference |
| `--device` | `cuda` | `cuda` / `cuda:N` / `cpu` |

### Output Files

```
outputs/
├── [name]_depth.png             # Colour-coded refined depth D₁
├── [name]_depth_raw.png         # uint16 metric depth (value/256 = metres)
├── [name]_reliability.png       # Reliability map (green = high trust)
├── [name]_uncertainty.png       # Aleatoric uncertainty σ̂
├── [name]_coarse.png            # Coarse depth D₀ (for D₀ vs D₁ inspection)
├── [name]_revision.png          # |D₁ − D₀| — where the revise stage acted
├── [name]_depth.npy             # float32 (H, W) metres
├── [name]_reliability.npy       # float32 (H, W) in (0,1)
├── [name]_uncertainty.npy       # float32 (H, W) metres
└── [name]_visualization.png     # 2×3 panel: RGB / sparse / D₀ / D₁ / R̂ / σ̂
```

### Python API

```python
import torch
from models.r3dc import R3DC
from utils.config import load_config, dict_to_namespace
from utils.checkpoint import load_ema_checkpoint

# --- Load ---
config = dict_to_namespace(load_config('configs/kitti.yaml'))
model  = R3DC(config)
model  = load_ema_checkpoint('checkpoints/model_best.pth.tar', model)
model.eval().cuda()

# --- Prepare input ---
rgb = load_image('image.png')                        # (1, 3, H, W) float32, normalized
sparse_depth, sparse_mask = load_sparse_depth('sparse.png')   # (1, 1, H, W)

# --- Forward ---
with torch.no_grad():
    predictions = model(rgb.cuda(), sparse_depth.cuda(), sparse_mask.cuda())

# --- Read outputs ---
depth       = predictions['d1_metric']     # refined metric depth  (1,1,H,W)
coarse      = predictions['d0_metric']     # pre-refinement depth
reliability = predictions['reliability']   # per-pixel reliability (0,1)
uncertainty = predictions['uncertainty']   # aleatoric σ̂ in metres
```

### Using Reliability Downstream

```python
# 1. Reject unreliable pixels before feeding a SLAM / planning stack
trusted = reliability > 0.6
depth_masked = torch.where(trusted, depth, torch.full_like(depth, float('nan')))

# 2. Weight a point cloud by trust
weights = reliability.flatten()

# 3. Turn reliability into an actionable flag
frame_trust = reliability.mean().item()
if frame_trust < 0.35:
    logger.warning("Low-trust frame — consider falling back to the previous keyframe")

# 4. Combine both heads: epistemic gate × aleatoric scale
effective_sigma = uncertainty / reliability.clamp(min=1e-3)
```

### ONNX Export & Latency

```bash
python scripts/export_onnx.py \
    --config configs/kitti.yaml \
    --checkpoint checkpoints/model_best.pth.tar \
    --output r3dc_base.onnx --opset 16

python scripts/benchmark_latency.py --onnx r3dc_base.onnx --iters 200
```

| Model | Resolution | Device | Latency | FPS | Peak VRAM |
|---|---|---|---:|---:|---:|
| `r3dc-base` (1.95 M) | 352×1216 | RTX 3090 | 28 ms | 35.7 | 1.8 GB |
| `r3dc-base` fp16 | 352×1216 | RTX 3090 | 17 ms | 58.8 | 1.1 GB |
| `r3dc-plus` (11.22 M) | 352×1216 | RTX 3090 | 61 ms | 16.4 | 3.4 GB |
| `r3dc-base` | 384×640 | Jetson Orin NX | 96 ms | 10.4 | 1.3 GB |

> Numbers are indicative for the reference environment; re-run `benchmark_latency.py` on your own hardware before quoting them.

---

## ⚙️ Configuration Reference

Every experiment is a single YAML file. Fields below are the complete surface area.

<details open>
<summary><b>Full annotated <code>configs/kitti.yaml</code></b></summary>

```yaml
# =====================================================================
# configs/kitti.yaml — R³DC on KITTI Depth Completion
# =====================================================================

model:
  # ---- Architecture ----
  base_width: 64                  # Base channel width; controls capacity (16/32/64)
  num_propagation_steps: 6        # CSPN++ iterations T
  kernel_size: 3                  # Propagation neighbourhood (3 → 8 neighbours)
  max_tokens: 512                 # Attention token budget (memory/accuracy knob)
  num_heads: 8                    # Transformer bottleneck heads
  dropout_path_rate: 0.1          # Stochastic depth
  group_norm_groups: 32           # GroupNorm groups
  center_weight: 0.2              # Fixed CSPN++ self-weight w_c
  use_dcn: true                   # DCNv2 in the sparse-depth encoder
  use_cbam: true                  # Channel+spatial attention

  # ---- Depth normalization ----
  d_min: 0.0                      # Minimum depth (metres)
  d_max: 80.0                     # Maximum depth (metres)
  epsilon: 0.001                  # Log-normalization epsilon

  # ---- Outputs ----
  predict_reliability: true       # Enable the reliability head + gate
  predict_uncertainty: true       # Enable the aleatoric head
  reliability_gate: true          # false → ablation: heads predict but do not steer
  dirichlet_anchors: true         # Re-clamp measured pixels each propagation step

loss:
  # ---- Weights ----
  weight_silog: 1.0
  weight_focal_berhu: 0.6
  weight_ssim: 0.2
  weight_anchor: 0.15
  weight_vnl: 0.1
  weight_dnc: 0.05
  weight_grad: 0.05
  weight_unc: 0.05
  weight_aux: 0.1

  # ---- Focal-BerHu ----
  focal_gamma: 2.0
  berhu_c_factor: 0.2

  # ---- SSIM ----
  ssim_window_size: 7

  # ---- VNL ----
  num_triplets: 1024

  # ---- Stage weighting ----
  coarse_weight: 0.5              # Weight on D₀ losses
  refined_weight: 1.0             # Weight on D₁ losses (carries the gate gradient)

optimizer:
  lr: 1.0e-4
  beta1: 0.9
  beta2: 0.999
  weight_decay: 1.0e-4
  scheduler: cosine_warm_restart  # cosine | cosine_warm_restart | step | onecycle
  warmup_epochs: 0
  t0: 10                          # First restart period (epochs)
  t_mult: 2                       # Period multiplier after each restart
  eta_min: 1.0e-6

dataset:
  dataset_name: kitti             # kitti | nyu | visdrone | drone_videos
  data_root: ./data
  split: train
  input_height: 352
  input_width: 1216
  batch_size: 4                   # Per-GPU under DDP
  num_workers: 4
  pin_memory: true
  shuffle: true
  drop_last: true

  # ---- Augmentation ----
  use_augmentation: true
  horizontal_flip: true
  color_jitter: true
  gamma_adjust: true
  sparse_dropout: 0.3             # Randomly drop 30% of sparse points
  cutmix_prob: 0.3

training:
  epochs: 20
  gradient_clip: 1.0
  log_interval: 10
  eval_interval: 1
  save_interval: 1
  checkpoint_dir: ./checkpoints
  log_dir: ./logs
  device: cuda
  seed: 42
  ema_decay: 0.9999
  use_amp: true
  use_ddp: false
  use_wandb: false
```

</details>

### Parameter Cheat-Sheet

| Group | Key | Type | Default | Effect if increased |
|---|---|---|---|---|
| model | `base_width` | int | 64 | Capacity ↑, params ~quadratic ↑, latency ↑ |
| model | `num_propagation_steps` | int | 6 | Smoother, better-filled depth; > 10 blurs edges |
| model | `kernel_size` | int | 3 | Wider propagation, cost ~`k²` |
| model | `max_tokens` | int | 512 | Attention fidelity ↑, VRAM ↑ |
| model | `center_weight` | float | 0.2 | Higher = more conservative refinement |
| loss | `weight_silog` | float | 1.0 | Dominates scale consistency |
| loss | `weight_anchor` | float | 0.15 | Higher = tighter fit to sparse points, risk of speckle |
| loss | `weight_vnl` | float | 0.1 | Higher = flatter planes, softer detail |
| dataset | `sparse_dropout` | float | 0.3 | Robustness to sensor degradation ↑, convergence slower |
| optimizer | `t0` / `t_mult` | int | 10 / 2 | Longer periods between LR restarts |
| training | `ema_decay` | float | 0.9999 | Smoother eval curves, slower to track improvements |

### Overriding from the Command Line

```bash
# Dotted overrides are merged over the YAML
python train.py --config configs/kitti.yaml \
    --set model.base_width=32 \
          dataset.batch_size=8 \
          optimizer.lr=2e-4 \
          training.epochs=12
```

---

## 📊 Results

> All checkpoints below use EMA weights. KITTI values are in **millimetres** (leaderboard convention); NYU and aerial values are in **metres**.

### KITTI Depth Completion — Official Benchmark

| Method | Params | RMSE (mm) ↓ | MAE (mm) ↓ | iRMSE ↓ | iMAE ↓ | Reliability |
|---|---:|---:|---:|---:|---:|:--:|
| CSPN++ | 17.4 M | 743.7 | 209.3 | 2.07 | 0.90 | ❌ |
| NLSPN | 26.8 M | 741.7 | 199.6 | 1.99 | 0.84 | ❌ |
| PENet | 131.0 M | 730.1 | 210.4 | 2.17 | 0.94 | ❌ |
| CompletionFormer | 12.7 M | **708.2** | 203.1 | 2.01 | 0.86 | ❌ |
| **R³DC (base)** | **1.95 M** | 786.4 | 221.7 | 2.18 | 0.95 | ✅ |
| **R³DC+ (extended)** | 11.22 M | 729.1 | 204.8 | 2.06 | 0.88 | ✅ |

**Reading the table.** R³DC-base gives up ~11 % RMSE relative to CompletionFormer while using **6.5× fewer parameters** and adding reliability. R³DC+ closes the gap to ~3 % with 1.5 M fewer parameters than CompletionFormer, and remains the only entry with a consumed reliability signal.

### NYU Depth V2

| Method | Params | δ₁ ↑ | AbsRel ↓ | RMSE (m) ↓ | MAE (m) ↓ |
|---|---:|---:|---:|---:|---:|
| CSPN++ | 17.4 M | 0.883 | 0.118 | 0.468 | 0.341 |
| NLSPN | 26.8 M | 0.891 | 0.112 | 0.441 | 0.318 |
| CompletionFormer | 12.7 M | 0.908 | 0.101 | 0.394 | 0.278 |
| DA-V2 backbone (frozen) | 94.6 M | 0.919 | 0.098 | 0.365 | 0.241 |
| **R³DC + ICH (ours)** | 94.6 M + **16.6 K trained** | **0.927** | **0.090** | **0.353** | **0.241** |

### VisDrone (synthetic GT)

| Method | RMSE (m) ↓ | δ₁ ↑ |
|---|---:|---:|
| Bilinear interpolation | 9.74 | 0.241 |
| CSPN++ | 5.18 | 0.631 |
| NLSPN | 4.91 | 0.664 |
| CompletionFormer | 4.34 | 0.712 |
| **R³DC+ (ours)** | **2.33** | **0.928** |

### RADI Comparison (NYU) — Reliability Sources

| Reliability source | Cost | REC ↑ | RBS ↑ | CAL ↓ | AUSE ↓ |
|---|---|---:|---:|---:|---:|
| Uniform (0.5) | free | 0.000 | 0.0 % | 0.248 | 0.184 |
| Inverse-gradient heuristic | free | 0.147 | 0.0 % | 0.183 | 0.139 |
| Depth-error proxy | free | 0.214 | 0.0 % | 0.151 | 0.117 |
| MC Dropout (20 passes) | 20× | 0.271 | 0.0 % | 0.126 | 0.098 |
| Deep Ensemble (3 models) | 3× | 0.303 | 0.0 % | 0.097 | 0.087 |
| **Learned R̂ (ours)** | **1×** | **0.371** | **41.3 %** | **0.041** | **0.061** |

**The RBS column is the point of the paper.** Every baseline scores exactly 0 % because their confidence is never consumed — however well-calibrated it may be, it does not change a single depth value. R³DC converts trust into a 41.3 % error reduction on the pixels it flags.

---

## 🔬 Ablation Studies

### Core Design Choices (KITTI val)

| # | Configuration | RMSE (mm) ↓ | REC ↑ | RBS ↑ | Δ RMSE |
|---|---|---:|---:|---:|---:|
| 0 | **Full R³DC+** | **729.1** | **0.371** | **41.3 %** | — |
| 1 | − reliability gate (heads kept, gate removed) | 761.8 | 0.089 | 0.0 % | +32.7 |
| 2 | − Dirichlet anchors | 748.3 | 0.352 | 38.1 % | +19.2 |
| 3 | − CMA (concat fusion instead) | 754.6 | 0.331 | 35.7 % | +25.5 |
| 4 | − DCNv2 in depth encoder | 745.9 | 0.344 | 37.2 % | +16.8 |
| 5 | − transformer bottleneck | 741.2 | 0.358 | 39.4 % | +12.1 |
| 6 | − uncertainty head | 733.7 | 0.366 | 40.8 % | +4.6 |
| 7 | − deep supervision | 738.5 | 0.361 | 40.1 % | +9.4 |
| 8 | − EMA at inference | 736.2 | 0.368 | 41.0 % | +7.1 |

Row 1 is the load-bearing ablation: removing the gate while keeping the reliability head costs 32.7 mm of RMSE **and** collapses REC from 0.371 to 0.089 — reliability only becomes meaningful when it is consumed.

### Propagation Steps `T`

| `T` | RMSE (mm) ↓ | RBS ↑ | Latency (ms) |
|---:|---:|---:|---:|
| 0 (no refinement) | 812.4 | — | 41 |
| 2 | 762.1 | 24.6 % | 47 |
| 4 | 738.9 | 36.2 % | 54 |
| **6** | **729.1** | **41.3 %** | **61** |
| 8 | 728.4 | 41.9 % | 68 |
| 12 | 734.7 | 40.2 % | 82 |

Returns saturate at `T = 6`; beyond `T = 8` over-smoothing begins to erase thin structures.

### Center Weight `w_c`

| `w_c` | RMSE (mm) ↓ | Stability |
|---:|---:|---|
| 0.0 | diverged | Unstable — no self-retention |
| 0.1 | 741.6 | Occasional NaN under AMP |
| **0.2** | **729.1** | Stable |
| 0.4 | 745.3 | Under-refinement |
| 0.6 | 768.9 | Refinement barely acts |

### Loss Term Contributions

| Removed term | RMSE (mm) ↓ | δ₁ ↑ | Note |
|---|---:|---:|---|
| none (full) | 729.1 | 0.920 | — |
| − SILog | 782.4 | 0.891 | Largest single contributor |
| − Focal-BerHu | 751.3 | 0.905 | Hurts hard-pixel accuracy |
| − SSIM | 740.8 | 0.914 | Softer structure |
| − Anchor | 758.6 | 0.902 | Drifts off measured points |
| − VNL | 738.2 | 0.916 | Planar regions warp |
| − Uncertainty | 733.7 | 0.918 | Minor on depth, removes σ̂ output |

### Sparse-Density Robustness (KITTI val, `r3dc-plus`)

| Retained LiDAR points | RMSE (mm) ↓ | Mean R̂ | REC ↑ |
|---|---:|---:|---:|
| 100 % | 729.1 | 0.68 | 0.371 |
| 50 % | 771.4 | 0.61 | 0.383 |
| 25 % | 843.7 | 0.53 | 0.401 |
| 10 % | 987.2 | 0.44 | 0.412 |
| 1 % | 1412.5 | 0.29 | 0.398 |

Reliability degrades gracefully and *honestly*: as measurements are removed the mean reliability drops and REC actually strengthens — the model becomes better at knowing that it does not know.

---

## 📐 RADI Protocol

**RADI** (*Reliability-Aware Depth Index*) evaluates a reliability map along three orthogonal axes. A method can pass one and fail the others, which is exactly why single-number confidence evaluations are misleading.

| Axis | Metric | Question |
|---|---|---|
| **Discrimination** | REC | Does reliability rank pixels by their true error? |
| **Utility** | RBS | Does reliability actually improve the output? |
| **Calibration** | CAL (+ AUSE) | Are the numeric values meaningful, not just ordinally correct? |

### Formal Definitions

Let `e_p = |d̂_p − d*_p|` be the absolute error at pixel `p` over the valid set `𝒱`.

**REC — Reliability–Error Correlation**

```
REC = − ρ_Spearman( R̂_p , e_p )        for p ∈ 𝒱
```

Spearman (rank) correlation is used rather than Pearson because reliability is only meaningful up to a monotone transform. The sign is flipped so that **higher is better**. A two-sided permutation p-value is reported alongside it (`radi_*_rec_p_value`).

**RBS — Revision Benefit Score**

```
𝒫_low = { p ∈ 𝒱 : R̂_p < median(R̂) }

RBS = 100 · ( E₀ − E₁ ) / E₀ ,    E₀ = mean_{p ∈ 𝒫_low} |D₀_p − d*_p|
                                  E₁ = mean_{p ∈ 𝒫_low} |D₁_p − d*_p|
```

RBS is the percentage error reduction achieved by the revise stage **on the pixels the model itself flagged as unreliable**. Any method whose confidence is not consumed scores exactly `0 %` by construction — `D₀ ≡ D₁`.

**CAL — Calibration Error**

Bin pixels into `B = 15` equal-width reliability bins. A pixel is *correct* if its relative error is within tolerance `τ = 0.10`:

```
acc(p) = 1[ |d̂_p − d*_p| / d*_p ≤ τ ]

CAL = Σ_{b=1..B}  (|𝔅_b| / |𝒱|) · | mean_{p ∈ 𝔅_b} acc(p)  −  mean_{p ∈ 𝔅_b} R̂_p |
```

This is an ECE over reliability bins. Perfect calibration means "pixels with reliability 0.8 are correct 80 % of the time".

**AUSE — Area Under the Sparsification Error curve**

Progressively remove the least reliable pixels and track the RMSE of the remainder. Compare this curve against the *oracle* curve obtained by removing the highest-error pixels:

```
AUSE = ∫₀¹ ( RMSE_R̂(f) − RMSE_oracle(f) ) df
```

where `f` is the removed fraction. `0` means the reliability ordering is oracle-equivalent.

### Region Masks

RADI reports each metric globally and over four difficulty-stratified regions, because global averages hide failures on exactly the pixels that matter.

| Region | Definition | Why it matters |
|---|---|---|
| **All** | Full valid set | Global baseline |
| **Edge** | Sobel magnitude of GT depth `> 0.05` | Depth discontinuities — where propagation methods bleed |
| **Textureless** | Local RGB std `< 8` (7×7 window) | No photometric cue to guide completion |
| **Far-depth** | `d* > 0.75 · d_max` | Sparse returns, largest absolute errors |

### Computing RADI

```python
from metrics.radi import RADI

radi = RADI(num_calibration_bins=15, tolerance=0.10)

results = radi.compute_radi(
    predictions={
        'reliability': reliability_map,   # (B, 1, H, W) in (0,1)
        'd0':          coarse_depth,      # (B, 1, H, W) normalized
        'd1':          refined_depth,     # (B, 1, H, W) normalized
        'd0_metric':   coarse_depth_metric,
        'd1_metric':   refined_depth_metric,
    },
    targets={
        'depth':       ground_truth,      # (B, 1, H, W) metres
        'sparse_mask': sparse_mask,       # (B, 1, H, W) — measured pixels excluded
    },
    rgb=rgb_image,                        # needed for the textureless mask
)

for region in ['all', 'edge', 'textureless', 'far_depth']:
    m = results[region]
    print(f"{region:12s}  REC={m['rec']:+.3f} (p={m['rec_p_value']:.1e})  RBS={m['rbs']:5.1f}%")

print(f"Global CAL  = {results['global_cal']:.3f}")
print(f"Global AUSE = {results['global_ause']:.3f}")
```

### Interpretation Guide

| Metric | Range | Perfect | Good | Poor |
|---|---|---:|---:|---:|
| REC | `[−1, 1]` | 1.0 | > 0.30 | < 0.10 |
| RBS | `(−∞, 100 %]` | 100 % | > 30 % | < 10 % |
| CAL | `[0, 0.25]` | 0.0 | < 0.05 | > 0.15 |
| AUSE | `[0, ∞)` | 0.0 | < 0.10 | > 0.15 |

> **Negative RBS is a real and important failure mode.** It means the revise stage made the flagged pixels *worse* — usually a sign the gate is inverted or the affinity normalization is broken. RADI surfaces this; ordinary RMSE does not.

### Applying RADI to Your Own Model

RADI is model-agnostic. Any method that produces a confidence map can be scored:

```python
# Methods without a refinement stage: pass d0 == d1, RBS will correctly be 0%
results = radi.compute_radi(
    predictions={'reliability': my_confidence, 'd0': my_depth, 'd1': my_depth,
                 'd0_metric': my_depth_m, 'd1_metric': my_depth_m},
    targets={'depth': gt, 'sparse_mask': mask},
    rgb=rgb,
)
```

We encourage reporting REC/RBS/CAL alongside RMSE in future confidence-aware depth work — and we consider a paper that reports only calibration, with no utility measurement, to be reporting half a result.

---

## 🔁 Reproducibility

### Deterministic Runs

```bash
python train.py --config configs/kitti.yaml --seed 42 --deterministic
```

`--deterministic` sets `torch.use_deterministic_algorithms(True)`, `cudnn.benchmark=False`, and a fixed dataloader worker seeding function. Expect a 15–25 % slowdown.

### Environment Capture

```bash
python -m utils.env_report > env.txt   # torch/CUDA/cuDNN/driver/GPU/commit hash
```

Every checkpoint embeds the config, the git commit, and the environment report:

```python
ckpt = torch.load('checkpoints/model_best.pth.tar', map_location='cpu')
print(ckpt['config'])       # exact YAML used
print(ckpt['git_commit'])
print(ckpt['env'])
print(ckpt['metrics'])      # val metrics at save time
```

### Reference Environment

| Component | Version |
|---|---|
| OS | Ubuntu 22.04 LTS |
| Python | 3.10.12 |
| PyTorch | 2.1.0 |
| CUDA / cuDNN | 11.8 / 8.9 |
| GPU | NVIDIA A100 80 GB (KITTI/NYU), RTX 3090 (aerial) |
| Random seed | 42 |

### Known Non-Determinism

- `DCNv2` backward is non-deterministic on some CUDA versions; expect ≤ 0.3 mm RMSE variation between runs.
- AMP loss scaling can change the step at which warm restarts trigger; disable with `--no_amp` for bit-exact comparisons.
- Reported numbers are the mean of **3 seeds** (42, 1337, 2024); per-seed spread on KITTI RMSE is ±2.1 mm.

---

## 🔧 Troubleshooting

### Out of Memory

```yaml
dataset:
  batch_size: 2              # halve it first
model:
  max_tokens: 256            # biggest single VRAM lever
training:
  use_amp: true
  accumulation_steps: 2      # keep the effective batch size
  gradient_checkpointing: true
```

```bash
# Diagnose
python -c "import torch; print(torch.cuda.memory_summary())"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Slow Training

| Cause | Fix |
|---|---|
| AMP disabled | `training.use_amp: true` |
| Dataloader starvation | `num_workers: 8`, `pin_memory: true`, `persistent_workers: true` |
| Full-resolution KITTI crops | `input_height: 256`, `input_width: 512` for development runs |
| DCN kernels un-fused | Rebuild DCNv2 against the installed CUDA toolkit |
| `--deterministic` left on | Remove it for production runs |

### Convergence Issues

```yaml
optimizer:
  lr: 5.0e-5
  warmup_epochs: 5
loss:
  weight_silog: 1.5
  weight_focal_berhu: 0.8
```

Also verify: `d_min`/`d_max` match your dataset (a wrong `d_max` silently destroys the log normalization), and invalid depths are `0`, not `NaN`.

### NaN Loss

```yaml
optimizer:
  lr: 1.0e-5
training:
  gradient_clip: 0.5
model:
  center_weight: 0.3         # w_c below 0.1 is unstable under AMP
```

Checklist:

1. `assert torch.isfinite(depth[mask]).all()` in the loader.
2. Confirm no GT pixel is exactly `0` inside the valid mask (log of zero).
3. Disable AMP for one epoch — if NaNs vanish, it is a loss-scaling problem, not a data problem.
4. Check `L_unc`: `σ̂` collapsing toward 0 produces `log σ̂ → −∞`; the Softplus head should have a `min_sigma` floor of `1e-3`.

### <a name="reliability-collapse"></a>Reliability Collapse

**Symptom.** `reliability_std → 0`, `REC ≈ 0`, `RBS ≈ 0` — the map is a constant.

| Cause | Fix |
|---|---|
| Gate wired backwards (`R̂_p` in the numerator) | Verify `Ã_{p→q} ∝ R̂_q`, i.e. the **neighbour's** trust |
| `refined_weight` too small | The gate gradient is the only reliability signal — keep `refined_weight ≥ 1.0` |
| Sigmoid saturated at init | Initialize the reliability head bias to `0.0` so `R̂ ≈ 0.5` at step 0 |
| Refinement effectively disabled | `num_propagation_steps ≥ 4` and `center_weight ≤ 0.3` |
| LR too high early | Add `warmup_epochs: 2` |

```bash
# Confirm the gate carries gradient
python -m tests.test_cspn_stability -k gate_gradient
```

### Depth Looks Right but Metrics Are Terrible

Almost always a units problem:

- KITTI PNG → metres requires `/256.0`, not `/1000.0`.
- Evaluate on `d1_metric`, never on the normalized `d1`.
- The KITTI leaderboard is in **millimetres**; the CLI prints **metres**.
- `sparse_mask` must exclude measured pixels from the RADI valid set, or REC is inflated.

### Checkpoint Loading Errors

```python
# Shape mismatch after changing base_width
model.load_state_dict(ckpt['state_dict'], strict=False)   # then inspect missing/unexpected

# EMA-only checkpoint
from utils.checkpoint import load_ema_checkpoint
model = load_ema_checkpoint(path, model)

# DDP-saved checkpoint on a single GPU (keys prefixed 'module.')
state = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
```

### Debugging Utilities

```python
# Verbose logging
import logging; logging.basicConfig(level=logging.DEBUG)

# GPU memory
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Reserved : {torch.cuda.memory_reserved()/1024**3:.2f} GB")

# Anomaly detection (slow — debugging only)
torch.autograd.set_detect_anomaly(True)
```

```python
# Visualize intermediate feature maps
import matplotlib.pyplot as plt

def visualize_features(features):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, (name, feat) in enumerate(features.items()):
        if i < 8:
            ax = axes[i // 4, i % 4]
            ax.imshow(feat[0, 0].detach().cpu().numpy())
            ax.set_title(name); ax.axis('off')
    plt.tight_layout(); plt.show()
```

---

## ❓ FAQ

<details>
<summary><b>What is the difference between reliability and uncertainty here?</b></summary>

`σ̂` (uncertainty) estimates **aleatoric** observation noise in metres — irreducible sensor/scene noise, trained with a Laplacian NLL. `R̂` (reliability) is a **relative, unitless gate** in `(0,1)` that answers "should this pixel's value be propagated to its neighbours, or overwritten by them?" It is closer to an epistemic trust score and is learned without any explicit target. They are complementary: `σ̂ / R̂` is a useful combined risk score.
</details>

<details>
<summary><b>Why does R³DC-base lose to CompletionFormer on KITTI RMSE?</b></summary>

Because it has 6.5× fewer parameters. We report it anyway. The base model exists for edge deployment where 1.95 M parameters and 28 ms latency are hard constraints; R³DC+ is the accuracy-oriented operating point. Suppressing the unfavourable row would misrepresent the trade-off.
</details>

<details>
<summary><b>Can I use R³DC without any sparse depth?</b></summary>

Yes — pass an all-zero mask and the model degrades to monocular relative depth with reliability. Accuracy drops substantially (see the sparse-density ablation at 1 %), and the reliability map correctly reports this by dropping to a mean around 0.29. It is not a replacement for a monocular foundation model, but it fails loudly rather than silently.
</details>

<details>
<summary><b>Does the reliability gate slow inference down?</b></summary>

Negligibly. The gate is an elementwise rescaling of an affinity tensor the network already computes — under 1 ms at 352×1216. Unlike MC-Dropout (20×) or ensembling (3×), reliability here is free at inference time.
</details>

<details>
<summary><b>How do I add a new dataset?</b></summary>

1. Subclass `datasets.base.BaseDepthDataset`, implement `__len__` and `__getitem__` returning `{'rgb', 'sparse_depth', 'sparse_mask', 'depth'}`.
2. Register it in `datasets/__init__.py`.
3. Copy a YAML and set `dataset_name`, `d_min`, `d_max`, `input_height`, `input_width`.
4. Run `python train.py --config configs/yours.yaml --overfit_batches 1` to confirm shapes and normalization before a full run.
</details>

<details>
<summary><b>Can RADI be applied to methods other than depth completion?</b></summary>

REC and CAL transfer to any dense regression task with a confidence map. RBS specifically requires a two-stage predict-then-revise structure, since it measures the delta between a pre- and post-refinement output. For single-stage methods, report REC/CAL/AUSE and state that RBS is not applicable rather than reporting `0 %` as if it were a score.
</details>

<details>
<summary><b>Why is the center weight fixed at 0.2 instead of learned?</b></summary>

We tried learning it. It drifts toward values that make propagation either explosive (`w_c → 0`) or inert (`w_c → 1`), and the loss landscape rewards the inert solution early in training. Fixing it at 0.2 is the single most important stability decision in the refinement stage (see the `w_c` ablation).
</details>

<details>
<summary><b>Are the aerial results comparable to KITTI/NYU?</b></summary>

No. VisDrone and Drone-Videos have no real depth ground truth; both the GT and the sparse prior are synthesized by our physics-motivated generator. Those experiments test whether the reliability mechanism transfers across imaging geometry, not absolute metric accuracy. Please do not cite the 2.33 m VisDrone RMSE as a comparison against real-sensor benchmarks.
</details>

<details>
<summary><b>What hardware do I need to reproduce the paper?</b></summary>

A single 24 GB GPU reproduces every experiment, at roughly 3× the wall-clock of the A100 reference runs. An 11 GB card works with `batch_size: 2` and `max_tokens: 256`, with a small accuracy penalty (~6 mm RMSE on KITTI).
</details>

---

## 🗺️ Roadmap

- [x] Reliability-gated CSPN++ refinement
- [x] RADI protocol (REC / RBS / CAL / AUSE) with region stratification
- [x] Cross-domain training recipe (KITTI / NYU / VisDrone / Drone-Videos)
- [x] Indoor Calibration Head for frozen foundation backbones
- [x] ONNX export + latency benchmarks
- [ ] Temporal R³DC — reliability propagation across video frames
- [ ] TensorRT / Jetson deployment guide
- [ ] Reliability-guided active sensing (where should the next LiDAR beam go?)
- [ ] Pretrained checkpoints for all four domains on HuggingFace Hub
- [ ] RADI as a standalone `pip install radi-metrics` package
- [ ] Extended journal version with additional real-sensor aerial evaluation

---

## 🤝 Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
# 1. Fork and branch
git checkout -b feature/my-improvement

# 2. Install dev tooling
pip install -e ".[dev]"
pre-commit install

# 3. Make changes, then verify
black . && isort . && flake8 .
pytest tests/ -v

# 4. Push and open a PR
git push origin feature/my-improvement
```

**Guidelines**

1. One logical change per PR; keep diffs reviewable.
2. New features need tests in `tests/`.
3. Any change touching the refinement stage must report RADI metrics before and after — a change that improves RMSE while collapsing RBS is a regression, not an improvement.
4. Follow `black` (line length 100) and type-annotate public functions.
5. Update this README when you change a CLI flag or a config key.

---

## 📝 Citation

If R³DC or the RADI protocol is useful in your work, please cite:

```bibtex
@inproceedings{mohammad2026r3dc,
  title     = {R\textsuperscript{3}DC: Reliability-Guided Reveal-to-Revise Depth
               Completion for Cross-Domain Sparse Perception},
  author    = {Mohammad, Noor Islam S. and Bayaz{\i}t, Ulu\u{g}},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
               Recognition (CVPR) Workshops --- 3D Geometry Generation for Scientific
               Computing},
  year      = {2026}
}
```

Thesis version:

```bibtex
@mastersthesis{mohammad2026r3dcthesis,
  title  = {R\textsuperscript{3}DC: A Unified Architecture for Reliability-Aware Depth
            Completion Across Heterogeneous Sensing Domains},
  author = {Mohammad, Noor Islam S.},
  school = {Istanbul Technical University},
  type   = {M.Sc. Thesis},
  year   = {2026}
}
```

If you use **RADI** specifically, please also state the version (`RADI v1.0`, `B = 15`, `τ = 0.10`) so numbers remain comparable across papers.

---

## 📄 License

Released under the **MIT License** — see [LICENSE](LICENSE).

Dataset licenses are separate and remain with their owners: KITTI (CC BY-NC-SA 3.0), NYU Depth V2 (research use), VisDrone (research use), Drone-Videos (per Kaggle terms). You are responsible for complying with them.

---

## 🙏 Acknowledgments

- **Istanbul Technical University**, Department of Computer Engineering, for research support and compute.
- **Prof. Dr. Uluğ Bayazıt** for supervision throughout this work.
- The authors of **CSPN++**, **NLSPN**, **PENet**, **CompletionFormer** and **Depth Anything V2**, whose public implementations made fair comparison possible.
- The **KITTI**, **NYU Depth V2**, **VisDrone** and **Drone-Videos** dataset providers.
- The open-source PyTorch ecosystem.

---

## 📧 Contact

| | |
|---|---|
| **Noor Islam S. Mohammad** | [mohammadn@itu.edu.tr](mailto:mohammadn@itu.edu.tr) |
| **Uluğ Bayazıt** | [bayazit@itu.edu.tr](mailto:bayazit@itu.edu.tr) |
| **Department** | Computer Engineering, Istanbul Technical University |
| **Issues** | [GitHub Issues](https://github.com/yourusername/r3dc/issues) — bugs, feature requests |
| **Discussions** | [GitHub Discussions](https://github.com/yourusername/r3dc/discussions) — usage questions |

---

## 📚 References

1. Uhrig, J. et al. *Sparsity Invariant CNNs.* 3DV, 2017. (KITTI Depth Completion benchmark)
2. Silberman, N. et al. *Indoor Segmentation and Support Inference from RGBD Images.* ECCV, 2012. (NYU Depth V2)
3. Cheng, X. et al. *CSPN++: Learning Context and Resource Aware Convolutional Spatial Propagation Networks for Depth Completion.* AAAI, 2020.
4. Park, J. et al. *Non-Local Spatial Propagation Network for Depth Completion.* ECCV, 2020.
5. Hu, M. et al. *PENet: Towards Precise and Efficient Image Guided Depth Completion.* ICRA, 2021.
6. Zhang, Y. et al. *CompletionFormer: Depth Completion with Convolutions and Vision Transformers.* CVPR, 2023.
7. Yang, L. et al. *Depth Anything V2.* NeurIPS, 2024.
8. Yin, W. et al. *Enforcing Geometric Constraints of Virtual Normal for Depth Prediction.* ICCV, 2019. (VNL)
9. Kendall, A. and Gal, Y. *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS, 2017.
10. Ilg, E. et al. *Uncertainty Estimates and Multi-Hypotheses Networks for Optical Flow.* ECCV, 2018. (AUSE / sparsification)
11. Guo, C. et al. *On Calibration of Modern Neural Networks.* ICML, 2017. (ECE)
12. Zhu, X. et al. *Deformable ConvNets v2: More Deformable, Better Results.* CVPR, 2019.
13. Woo, S. et al. *CBAM: Convolutional Block Attention Module.* ECCV, 2018.
14. Zhu, P. et al. *Detection and Tracking Meet Drones Challenge.* TPAMI, 2021. (VisDrone)

---

## 📌 Changelog

| Version | Date | Changes |
|---|---|---|
| `1.0.0` | 2026-06 | CVPR-W camera-ready release: full RADI protocol, 4-domain configs, ICH, ONNX export |
| `0.9.0` | 2026-04 | Indoor Calibration Head; NYU δ₁ = 0.927 |
| `0.8.0` | 2026-02 | Aerial synthetic-depth generator; VisDrone + Drone-Videos support |
| `0.7.0` | 2025-12 | Reliability-gated CSPN++; first positive RBS |
| `0.5.0` | 2025-10 | Baseline encoder–decoder with auxiliary reliability head (gate not yet closed) |

---

<div align="center">

**R³DC** — *Knowing what to revise.*

Built at Istanbul Technical University · [MIT Licensed](LICENSE) · Issues and PRs welcome

</div>
