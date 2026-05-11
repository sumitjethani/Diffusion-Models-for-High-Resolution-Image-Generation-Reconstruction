# Diffusion Models for High-Resolution Image Generation and Reconstruction

A from-scratch implementation of **DDPM (Denoising Diffusion Probabilistic Models)** with **DDIM sampling** for high-resolution face image generation and reconstruction, trained on the CelebA-HQ dataset.

---

## Demo

🤗 **Live Demo:** [HuggingFace Space](https://huggingface.co/spaces/Sumit-Jethani/Diffusion-Models-for-High-Resolution-Image-Generation-and-Reconstruction)

---

## What This Project Does

- **Image Generation** — Generate realistic human faces from pure Gaussian noise using DDIM sampling
- **Image Reconstruction** — Add noise to a target image up to timestep t, then denoise it back using the reverse diffusion process
- **Forward Diffusion Visualization** — Visualize how a clean image progressively becomes noise across timesteps
- **Reverse Diffusion Visualization** — Visualize how pure noise is denoised back into a clean image

---

## Model Architecture

### U-Net Backbone
- Encoder-Decoder structure with skip connections
- Channel progression: 64 → 128 → 256 (controlled by `CH_MULT`)
- 2 Residual Blocks per resolution level

### Key Components

**Sinusoidal Timestep Embeddings**
- Encodes the current diffusion timestep t into a continuous embedding
- Injected into every ResBlock via scale-shift conditioning

**Residual Block**
- GroupNorm + SiLU activation
- Time embedding projected to scale and shift (AdaGN conditioning)
- Skip connection handles channel mismatch via 1×1 Conv

**Memory-Efficient Attention Block**
- Multi-head self-attention at bottleneck resolution
- Uses PyTorch's `scaled_dot_product_attention` with Flash Attention support
- Reduces OOM errors on T4 GPUs

**Downsample / Upsample**
- Strided Conv2d for downsampling (learnable, no information loss)
- Nearest neighbor Upsample + Conv for upsampling (avoids checkerboard artifacts)

### EMA (Exponential Moving Average)
- Maintains a smoothed copy of model weights during training
- Used exclusively for inference — produces sharper and more stable samples

---

## Noise Schedule

### Cosine Schedule (default — improved DDPM)
```
f(t) = cos((t/T + 0.008) / 1.008 × π/2)²
beta_t = 1 - f(t) / f(t-1)
```
Cosine schedule keeps noise additions small early and late in the process, preventing over-noising at the end of the forward process. Produces better sample quality than linear schedule.

### Linear Schedule (baseline)
```
beta_t = linspace(beta_start, beta_end, T)
```

### Forward Process
```
q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
```
Using the reparameterization trick, any timestep `x_t` can be sampled directly from `x_0` in a single step.

---

## Sampling Methods

### DDPM Sampling (standard)
- Iterates through all T=400 timesteps
- Slower but theoretically exact
- Adds stochastic noise at each step

### DDIM Sampling (fast — default for inference)
- Skips timesteps using a deterministic subset (50 steps by default)
- `eta=0.0` for fully deterministic sampling
- Produces sharper results 8× faster than DDPM

---

## Training Details

| Parameter | Value |
|---|---|
| Dataset | CelebA-HQ 256 |
| Image Size | 128×128 |
| Diffusion Timesteps (T) | 400 |
| Noise Schedule | Cosine |
| Batch Size | 16 |
| Epochs | 50 (early stopping, patience=10) |
| Optimizer | AdamW (lr=2e-4, weight decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Gradient Clipping | 1.0 |
| Mixed Precision | AMP (torch.amp) |
| GPUs | Dual T4 (Kaggle) |
| Loss | MSE (predicted noise vs actual noise) |

---

## Loss Function

```
L = MSE(ε_θ(x_t, t), ε)
```

The model predicts the noise `ε` that was added to `x_0` to produce `x_t`. Training minimizes the mean squared error between predicted and actual noise. This is the simplified DDPM objective (Ho et al. 2020).

---

## Evaluation

| Metric | Score |
|---|---|
| PSNR | Evaluated on 5 reconstructed images |
| SSIM | Evaluated on 5 reconstructed images |

Reconstruction evaluated by: adding noise up to t=250, then denoising back with DDIM (50 steps), comparing result against original.

---

## Project Structure

```
├── Diffusion_Models_for_Image_generation_and_reconstruction.ipynb   # Full training notebook
├── app.py                          # Gradio app (HuggingFace deployment)
├── requirements.txt                # Dependencies
└── README.md
```

---

## Installation & Usage

```bash
# Clone the repo
git clone https://github.com/sumitjethani/Diffusion-Models-for-High-Resolution-Image-Generation-and-Reconstruction.git
cd Diffusion-Models-for-High-Resolution-Image-Generation-and-Reconstruction

# Install dependencies
pip install torch torchvision accelerate torchmetrics[image] einops gradio
```

---

## Dataset

[CelebA-HQ 256](https://www.kaggle.com/datasets/denislukovnikov/celebahq256-images-only) — Kaggle

---

## References

- [DDPM — Ho et al. 2020](https://arxiv.org/abs/2006.11239)
- [DDIM — Song et al. 2020](https://arxiv.org/abs/2010.02502)
- [Improved DDPM (Cosine Schedule) — Nichol & Dhariwal 2021](https://arxiv.org/abs/2102.09672)

---

## Author

**Sumit Jethani**
- GitHub: [github.com/sumitjethani](https://github.com/sumitjethani)
- LinkedIn: [linkedin.com/in/sumit-jethani](https://linkedin.com/in/sumit-jethani)
- HuggingFace: [huggingface.co/Sumit-Jethani](https://huggingface.co/Sumit-Jethani)
