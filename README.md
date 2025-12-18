# Stable Diffusion with Custom Guidance Loss

A PyTorch implementation of Stable Diffusion v1.4 enhanced with custom guidance losses for improved image generation quality. This project implements three complementary loss functions that guide the diffusion process to produce semantically accurate, sharp, and well-composed images.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **CLIP-Guided Generation**: Semantic alignment between generated images and text prompts
- **Frequency Balance Loss**: Reduces blur by balancing low and high-frequency components
- **Composition Loss**: Encourages centered and balanced image composition
- **Interactive Gradio Interface**: Real-time parameter tuning and generation
- **Multi-style Generation**: Supports various artistic styles (oil painting, watercolor, cyberpunk, etc.)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Stable Diffusion Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Text Prompt → CLIP Text Encoder → Text Embeddings          │
│                                                               │
│  Random Noise → UNet (50 steps) → Denoised Latents → VAE    │
│                    ↓                                          │
│              Custom Guidance                                  │
│            (Applied every 5 steps)                            │
│                    ↓                                          │
│         ┌──────────┴──────────┐                              │
│         │  Composite Loss     │                              │
│         ├─────────────────────┤                              │
│         │  • CLIP Loss        │  (w=1.0)                     │
│         │  • Frequency Loss   │  (w=0.15)                    │
│         │  • Composition Loss │  (w=0.05)                    │
│         └─────────────────────┘                              │
│                    ↓                                          │
│         Gradient-based latent update                          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Loss Functions

### 1. CLIP Semantic Loss (L_clip)

Ensures the generated image semantically aligns with the input text prompt using OpenAI's CLIP model.

**Mathematical Formulation:**
```
L_clip = 1 - cos_similarity(f_img(I), f_text(T))
```

**Implementation Details:**
- Uses CLIP ViT-Base-Patch32 model
- Images resized to 224×224 with CLIP preprocessing
- Normalized image and text features
- Weight: `w_clip = 1.0`

**Code Reference:** [app.py:94-111](app.py#L94-L111)

```python
# CLIP preprocessing
x = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
x = (x - mean) / std

# Feature extraction
img_feat = clip_model.get_image_features(pixel_values=x)
img_feat = F.normalize(img_feat, dim=-1)

# Cosine similarity loss
sim = (img_feat @ clip_text_feat.t()).squeeze(-1)
L_clip = (1.0 - sim).mean()
```

### 2. Frequency Balance Loss (L_freq)

Reduces blurriness by encouraging a balanced distribution between low and high-frequency components in the frequency domain.

**Mathematical Formulation:**
```
Luminance: Y = 0.2989R + 0.5870G + 0.1140B
FFT: F = FFT2D(Y)
Magnitude: M = |fftshift(F)|

Low_energy = mean(M[r ≤ 0.15])
High_energy = mean(M[r > 0.15])
Ratio = High_energy / (Low_energy + High_energy)

L_freq = |Ratio - 0.35|
```

**Implementation Details:**
- Converts RGB to luminance (Y channel)
- 2D FFT with frequency shift
- Splits frequency space at radius 0.15
- Target high-frequency ratio: 35%
- Weight: `w_freq = 0.15`

**Code Reference:** [app.py:113-139](app.py#L113-L139)

```python
# Convert to luminance
y = (0.2989 * images[:,0] + 0.5870 * images[:,1] + 0.1140 * images[:,2])

# FFT magnitude
Freq = torch.fft.fft2(y)
mag = torch.abs(torch.fft.fftshift(Freq, dim=(-2, -1)))

# Frequency separation
split = 0.15
low_mask  = (rr <= split).float()
high_mask = (rr > split).float()

low_e  = (mag * low_mask).mean(dim=(-2, -1))
high_e = (mag * high_mask).mean(dim=(-2, -1))
ratio = high_e / (low_e + high_e + 1e-8)

# Target ratio loss
target_ratio = 0.35
L_freq = (ratio - target_ratio).abs().mean()
```

### 3. Composition Loss (L_comp)

Encourages visually balanced compositions by keeping the center of saliency near the image center.

**Mathematical Formulation:**
```
Saliency: S(x,y) = √(∂Y/∂x)² + (∂Y/∂y)²

Center of mass:
cx = Σ(S(x,y) · x) / Σ(S(x,y))
cy = Σ(S(x,y) · y) / Σ(S(x,y))

L_comp = (cx - 0.5)² + (cy - 0.5)²
```

**Implementation Details:**
- Computes image gradients on luminance
- Saliency map from gradient magnitude
- Calculates center of mass
- Penalizes deviation from center (0.5, 0.5)
- Weight: `w_comp = 0.05`

**Code Reference:** [app.py:141-160](app.py#L141-L160)

```python
# Gradient-based saliency
y2 = y.unsqueeze(1)
gx = y2[..., :, 1:] - y2[..., :, :-1]
gy = y2[..., 1:, :] - y2[..., :-1, :]
sal = torch.sqrt(gx*gx + gy*gy + 1e-12).squeeze(1)

# Center of mass calculation
mass = sal.sum(dim=(-2, -1)) + 1e-8
cx = (sal * xs).sum(dim=(-2, -1)) / mass
cy = (sal * ys).sum(dim=(-2, -1)) / mass

# Distance from center
L_comp = ((cx - 0.5)**2 + (cy - 0.5)**2).mean()
```

### Composite Loss Function

The final loss combines all three components:

```
L_total = w_clip × L_clip + w_freq × L_freq + w_comp × L_comp
       = 1.0 × L_clip + 0.15 × L_freq + 0.05 × L_comp
```

## Guidance Mechanism

### Gradient-based Latent Updates

The custom guidance is applied during the denoising process:

**Code Reference:** [app.py:230-263](app.py#L230-L263)

```python
# Applied every 5 steps, after 20% of total steps
if (i % 5 == 0) and (i > num_inference_steps * 0.2):

    # Enable gradient computation
    latents = latents.detach().requires_grad_()

    # Predict x0 approximation
    latents_x0 = latents - sigma * noise_pred.detach()

    # Decode to image space
    denoised_images = vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5

    # Calculate composite loss
    loss = blue_loss(denoised_images.float(), clip_text_feat,
                     clip_weight=clip_scale,
                     freq_weight=freq_scale,
                     comp_weight=comp_scale)

    # Compute gradients
    cond_grad = torch.autograd.grad(loss, latents)[0]

    # Gradient clipping
    max_norm = 0.5
    g_norm = cond_grad.norm()
    cond_grad = cond_grad * (max_norm / (g_norm + 1e-8))

    # Update latents
    step_size = 0.05
    latents = latents.detach() - step_size * cond_grad * sigma**2
```

**Key Parameters:**
- **Guidance Frequency**: Every 5 steps
- **Start Point**: After 20% of total steps (to avoid corrupting early structure)
- **Gradient Clipping**: Max norm of 0.5 for stability
- **Step Size**: 0.05 (controls guidance strength)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/diffusion_model_code.git
cd diffusion_model_code

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
transformers>=4.25.1
diffusers>=0.20.0
gradio>=3.40.0
pillow>=9.0.0
numpy>=1.23.0
torchvision>=0.15.0
accelerate>=0.20.0
ftfy>=6.1.0
```

## Usage

### Gradio Web Interface

Launch the interactive web interface:

```bash
python app.py
```

The interface provides controls for:
- **Prompt**: Text description of desired image
- **Seed**: Random seed for reproducibility (-1 for random)
- **CLIP Guidance Scale**: Semantic alignment strength (0-1000, default: 200)
- **Frequency Guidance Scale**: Sharpness enhancement (0-500, default: 30)
- **Composition Guidance Scale**: Centered composition (0-500, default: 10)
- **Inference Steps**: Number of denoising steps (10-100, default: 50)
- **CFG Scale**: Classifier-free guidance strength (1-20, default: 8.0)

### Jupyter Notebook

Explore the implementation interactively:

```bash
jupyter notebook Stable_Diffusion_Deep_Dive.ipynb
```

### Programmatic Usage

```python
from app import generate

# Generate with default settings
image = generate(
    prompt="A futuristic city with neon lights",
    seed=42,
    clip_scale=200,
    freq_scale=30,
    comp_scale=10,
    num_inference_steps=50,
    guidance_scale=8.0
)

# Save the image
image.save("output.png")
```

## Examples

### Style Variations

The model supports various artistic styles by modifying the prompt:

```python
base_prompt = "A campfire with group of people"

styles = [
    "oil painting, impasto, thick brush strokes, canvas texture",
    "watercolor painting, soft wash, paper texture",
    "pencil sketch, graphite, cross-hatching, monochrome",
    "cyberpunk, neon lighting, futuristic, high contrast",
    "pixel art, 16-bit, low resolution, retro game style"
]
```

### Comparison: With vs Without Custom Guidance

| Configuration | CLIP Loss | Frequency Loss | Composition Loss | Result |
|--------------|-----------|----------------|------------------|---------|
| Baseline (Prompt Only) | ✗ | ✗ | ✗ | Good semantic match, may be blurry |
| With CLIP Guidance | ✓ (200) | ✗ | ✗ | Improved semantic accuracy |
| With Frequency | ✗ | ✓ (30) | ✗ | Sharper details |
| Full Guidance | ✓ (200) | ✓ (30) | ✓ (10) | Sharp, accurate, well-composed |

## Technical Implementation

### Model Components

1. **VAE (Variational Autoencoder)**
   - Model: CompVis/stable-diffusion-v1-4
   - Latent space: 4×64×64 for 512×512 images
   - Scaling factor: 0.18215

2. **Text Encoder**
   - Model: CLIP ViT-L/14 (openai/clip-vit-large-patch14)
   - Max sequence length: 77 tokens
   - Embedding dimension: 768

3. **UNet**
   - Architecture: UNet2DConditionModel
   - Cross-attention conditioning
   - Time embedding

4. **Scheduler**
   - Type: LMS Discrete Scheduler
   - Beta schedule: Scaled linear
   - Beta range: [0.00085, 0.012]
   - Training timesteps: 1000

### Classifier-Free Guidance (CFG)

Standard CFG is applied at each denoising step:

```python
noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
```

The custom guidance acts as an additional signal on top of CFG.

## Performance Considerations

### GPU Memory

- **Baseline**: ~6-8 GB VRAM (FP32)
- **With Guidance**: ~8-10 GB VRAM (due to gradient computation)

### Speed

- **Without Custom Guidance**: ~5-7 seconds per image (50 steps)
- **With Custom Guidance**: ~8-12 seconds per image (50 steps)
  - Additional ~40-50ms per guided step for loss computation and gradient update

### Device Support

- ✅ CUDA (NVIDIA GPUs)
- ✅ MPS (Apple Silicon M1/M2)
- ✅ CPU (slow, not recommended)

## Project Structure

```
diffusion_model_code/
├── app.py                              # Gradio web interface
├── Stable_Diffusion_Deep_Dive.ipynb   # Jupyter notebook implementation
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── learned_embeds.bin                  # Custom embeddings (optional)
```

## Key Implementation Files

- [app.py](app.py) - Main application with Gradio interface
- [app.py:87-162](app.py#L87-L162) - Complete `blue_loss` function
- [app.py:167-268](app.py#L167-L268) - Generation function with guidance
- [Stable_Diffusion_Deep_Dive.ipynb](Stable_Diffusion_Deep_Dive.ipynb) - Detailed exploration

## Hyperparameter Tuning Guide

### CLIP Guidance Scale
- **Low (0-100)**: Minimal semantic guidance, more creative freedom
- **Medium (100-300)**: Balanced guidance (recommended: 200)
- **High (300-1000)**: Strong semantic enforcement, less variation

### Frequency Guidance Scale
- **Low (0-20)**: Subtle sharpness improvement
- **Medium (20-50)**: Noticeable detail enhancement (recommended: 30)
- **High (50-500)**: Very sharp, may introduce artifacts

### Composition Guidance Scale
- **Low (0-10)**: Gentle centering (recommended: 10)
- **Medium (10-50)**: Strong centering bias
- **High (50-500)**: Very centered, may feel unnatural

## Limitations

- Guidance increases generation time by ~50-100%
- Requires more GPU memory for gradient computation
- Very high guidance scales can introduce artifacts
- Composition loss assumes centered subjects are desirable






