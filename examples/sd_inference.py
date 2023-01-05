import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, logging
from tqdm.auto import tqdm
from PIL import Image

logging.set_verbosity_error()
device = "cuda" if torch.cuda.is_available() else "cpu"

# This is a refactored version of the code from:
# https://github.com/fastai/diffusion-nbs/blob/master/Stable%20Diffusion%20Deep%20Dive.ipynb
# It demonstrates the text to image pipeline

def load_pretrained_models():
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    return vae, tokenizer, text_encoder, unet

def compute_text_embeddings(tokenizer, text_encoder, prompt):
    # We need uncondition because we are doing classifier-free guidance
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings

def initialize_scheduler(num_inference_steps):
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    scheduler.set_timesteps(num_inference_steps)
    return scheduler

def initialize_latents(batch_size, height, width, generator, unet, scheduler):
    latents = torch.randn(
      (batch_size, unet.in_channels, height // 8, width // 8),
      generator=generator,
    )
    latents = latents.to(device)
    latents = latents * scheduler.init_noise_sigma
    return latents

prompt = "A watercolor painting of an otter"
height = width = 512
batch_size = 1
num_inference_steps = 30
guidance_scale = 7.5
generator = torch.manual_seed(32)
vae, tokenizer, text_encoder, unet = load_pretrained_models()
text_embeddings = compute_text_embeddings(tokenizer, text_encoder, prompt)
scheduler = initialize_scheduler(num_inference_steps)
latents = initialize_latents(batch_size, height, width, generator, unet, scheduler)

with torch.autocast("cuda"):
    for i, t in tqdm(enumerate(scheduler.timesteps)):
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample

# Display
image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0].save('image.png')
