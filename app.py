import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from PIL import Image
import numpy as np
import gradio as gr
from tqdm.auto import tqdm
import torchvision.transforms as T
import torch.nn.functional as F
import os

# --- Setup ---
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# Fix for MPS fallbacks if needed
if device == "mps":
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

print(f"Using device: {device}")

# --- Global Models ---
# We load these once
vae = None
tokenizer = None
text_encoder = None
unet = None
scheduler = None
clip_model = None
clip_processor = None

def load_models():
    global vae, tokenizer, text_encoder, unet, scheduler, clip_model, clip_processor
    
    print("Loading Stable Diffusion models...")
    sd_model_id = "CompVis/stable-diffusion-v1-4"
    vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    unet = UNet2DConditionModel.from_pretrained(sd_model_id, subfolder="unet").to(device)
    
    # Scheduler setup from notebook
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085, 
        beta_end=0.012, 
        beta_schedule="scaled_linear", 
        num_train_timesteps=1000
    )
    
    print("Loading CLIP model for guidance...")
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    print("Models loaded successfully.")

# Load models on startup
load_models()

# --- Helper Functions ---

def set_timesteps(scheduler, num_inference_steps):
    scheduler.set_timesteps(num_inference_steps)
    # Minor fix for MPS compatibility mentioned in notebook
    scheduler.timesteps = scheduler.timesteps.to(torch.float32)

def pil_to_latent(input_im):
    # Not strictly used in generate_one but useful context
    with torch.no_grad():
        latent = vae.encode(T.ToTensor()(input_im).unsqueeze(0).to(device)*2-1) 
    return 0.18215 * latent.latent_dist.sample()

def latents_to_pil(latents):
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

# --- Loss Function ---

# --- Loss Function ---

def blue_loss(images, clip_text_feat, clip_weight=1.0, freq_weight=0.15, comp_weight=0.05):
    """
    images: (B,3,H,W) in range [0,1]
    clip_text_feat: target text embedding for CLIP loss
    Returns a single scalar loss.
    """
    # ----------------------------
    # (A) CLIP semantic loss
    # ----------------------------
    # Differentiable preprocess for CLIP
    # Resize to 224x224
    x = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)

    # CLIP normalization constants
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1,3,1,1)
    x = (x - mean) / std

    img_feat = clip_model.get_image_features(pixel_values=x)
    img_feat = F.normalize(img_feat, dim=-1)

    # 1 - cosine similarity (minimize => image matches prompt more)
    # clip_text_feat should be (1, D)
    sim = (img_feat @ clip_text_feat.t()).squeeze(-1)  # (B,)
    L_clip = (1.0 - sim).mean()

    # ----------------------------
    # (B) Frequency balance loss (reduce blur)
    # ----------------------------
    # luminance
    y = (0.2989 * images[:,0] + 0.5870 * images[:,1] + 0.1140 * images[:,2])  # (B,H,W)

    Freq = torch.fft.fft2(y)
    mag = torch.abs(torch.fft.fftshift(Freq, dim=(-2, -1)))  # (B,H,W)

    B, H, W = mag.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-0.5, 0.5, H, device=mag.device),
        torch.linspace(-0.5, 0.5, W, device=mag.device),
        indexing="ij"
    )
    rr = torch.sqrt(xx**2 + yy**2)

    split = 0.15  # center radius treated as "low frequency"
    low_mask  = (rr <= split).float()
    high_mask = (rr >  split).float()

    low_e  = (mag * low_mask).mean(dim=(-2, -1))
    high_e = (mag * high_mask).mean(dim=(-2, -1))
    ratio = high_e / (low_e + high_e + 1e-8)

    target_ratio = 0.35
    L_freq = (ratio - target_ratio).abs().mean()

    # ----------------------------
    # (C) Tiny composition loss (keep saliency near center)
    # ----------------------------
    # saliency via gradient magnitude on luminance
    y2 = y.unsqueeze(1)  # (B,1,H,W)
    gx = y2[..., :, 1:] - y2[..., :, :-1]
    gy = y2[..., 1:, :] - y2[..., :-1, :]
    gx = F.pad(gx, (0,1,0,0))
    gy = F.pad(gy, (0,0,0,1))
    sal = torch.sqrt(gx*gx + gy*gy + 1e-12).squeeze(1)  # (B,H,W)

    xs = torch.linspace(0, 1, W, device=sal.device).view(1,1,W).expand(B,H,W)
    ys = torch.linspace(0, 1, H, device=sal.device).view(1,H,1).expand(B,H,W)

    mass = sal.sum(dim=(-2, -1)) + 1e-8
    cx = (sal * xs).sum(dim=(-2, -1)) / mass
    cy = (sal * ys).sum(dim=(-2, -1)) / mass

    # target center (0.5,0.5)
    L_comp = ((cx - 0.5)**2 + (cy - 0.5)**2).mean()

    return (clip_weight * L_clip) + (freq_weight * L_freq) + (comp_weight * L_comp)


# --- Generation Function ---

def generate(prompt, seed, clip_scale, freq_scale, comp_scale, num_inference_steps=50, guidance_scale=8.0):
    if seed == -1:
        seed = torch.randint(0, 1000000, (1,)).item()
    
    print(f"Generating: '{prompt}', Seed: {seed}, Scales: CLIP={clip_scale}, Freq={freq_scale}, Comp={comp_scale}")
    
    with torch.no_grad():
        text_inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True, truncation=True).to(device)
        txt = clip_model.get_text_features(**text_inputs)
        clip_text_feat = F.normalize(txt, dim=-1) # (1, D)

    # MPS Generator Fix:
    # On MPS, explicitly passing a generator to torch.randn can cause issues if the generator is not perfectly compatible.
    # Safe bet: use the global torch random state seeded manually.
    generator = None
    if device == "mps":
        torch.manual_seed(int(seed))
    else:
        generator = torch.Generator(device=device).manual_seed(int(seed))
    
    batch_size = 1
    height = 512
    width = 512
    
    # Prep text for UNet
    text_input = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        
    # Uncond input
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
        
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    # Prep Scheduler
    set_timesteps(scheduler, num_inference_steps)
    
    # Prep latents
    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device=device
    )
    latents = latents * scheduler.init_noise_sigma
    
    # Loop
    for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
        # expand latents for CFG
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
        # Perform CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # --- ADDITIONAL GUIDANCE (Blue Loss) ---
        # Apply if any scale is > 0
        if (clip_scale > 0 or freq_scale > 0 or comp_scale > 0) and (i % 5 == 0) and (i > num_inference_steps * 0.2):
            
            latents = latents.detach().requires_grad_()
            
            # Predict x0 approximation
            latents_x0 = latents - sigma * noise_pred.detach()
            
            # Decode to image space
            denoised_images = vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5 
            denoised_images = denoised_images.clamp(0, 1)
            
            # Calculate loss using the separate scales
            # The 'scale' here is effectively the weight (since we don't need a separate master scale)
            loss = blue_loss(denoised_images.float(), clip_text_feat, 
                             clip_weight=clip_scale, 
                             freq_weight=freq_scale, 
                             comp_weight=comp_scale)
            
            if i % 10 == 0:
                print(f"Step {i}, Guided Loss: {loss.item()}")
                
            # Get gradient
            cond_grad = torch.autograd.grad(loss, latents, retain_graph=False, create_graph=False)[0]
            
            # Gradient clipping
            max_norm = 0.5
            g_norm = cond_grad.norm()
            cond_grad = cond_grad * (max_norm / (g_norm + 1e-8))
            
            # Modify latents
            step_size = 0.05
            latents = latents.detach() - step_size * cond_grad * sigma**2
            
        # Step with scheduler
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        
    return latents_to_pil(latents)[0]


# --- Gradio UI ---

title = "Stable Diffusion with Custom Guidance"
description = """
Generate images using Stable Diffusion with custom guidance losses.
Adjust the scales below to control the influence of each loss component.
"""

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", value="A futuristic city with neon lights"),
        gr.Number(label="Seed (-1 for random)", value=42),
        gr.Slider(label="CLIP Guidance Scale", minimum=0, maximum=1000, value=200, step=10, 
                  info="Keeps image semantically aligned with prompt."),
        gr.Slider(label="Frequency Guidance Scale", minimum=0, maximum=500, value=30, step=5, 
                  info="Discourages blurriness."),
        gr.Slider(label="Composition Guidance Scale", minimum=0, maximum=500, value=10, step=5,
                  info="Encourages centered saliency."),
        gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=50, step=1),
        gr.Slider(label="Guidance Scale (CFG)", minimum=1, maximum=20, value=8.0, step=0.5),
    ],
    outputs=gr.Image(label="Generated Image"),
    title=title,
    description=description
)

if __name__ == "__main__":
    demo.launch(share=True)
