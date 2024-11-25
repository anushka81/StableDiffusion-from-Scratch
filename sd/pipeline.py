import torch 
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512 
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8

def generate_image( prompt,
    uncond_prompt=None, # negative prompt
    input_image=None,
    strength=0.8, # how much noise we want to add, more the noise less the output ressembles the input image.
    do_cfg=True, # classifier free guidance, for eg - model to output two outputs, one with the prompt, and one without the prompt
    cfg_scale=7.5, # how much attention to pay to prompt
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,):
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("strength must be between 1 and 0")
        if idle_device:
            to_idle : lambda x : x.to(idle_device)
        else:
            to_idle = lambda x : x
        
        generator = torch.Generator(device=device)
        if seed is not None:
           generator.manual_seed(seed)
        else:
            generator.seed()

        clip = models["clip"]
        clip.to(device)

        # classifier free guidance
        if do_cfg:
            # convert the prompt into tokens using the tokenizer
            conditional_tokens = tokenizer.batch_encode_plus(
                [prompt],
                padding="max_length",
                max_length=77,
            ).input_ids

            #(Batch_Size, Sequence_Length)
            conditional_tokens = torch.tensor(conditional_tokens, dtype=torch.long, device=device).to(device)
            #(Batch_Size, Sequence_Length) -> (Batch_Size, Sequence_Length, dimen)
            conditional_context = clip(conditional_tokens)

            # convert the uncond_prompt into tokens using the tokenizer
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt],
                padding="max_length",
                max_length=77,
            ).input_ids

            #(Batch_Size, Sequence_Length)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device).to(device)
            #(Batch_Size, Sequence_Length) -> (Batch_Size, Sequence_Length, dimen)
            uncond_context = clip(uncond_tokens)

            #(2, seq_len, dimen) = (2, 77, 768)
            context = torch.cat([conditional_context, uncond_context])
        else:
             # convert the prompt into tokens using the tokenizer
            tokens = tokenizer.batch_encode_plus(
                [prompt],
                padding="max_length",
                max_length=77,
            ).input_ids

            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

            #(1, 77, 768)
            context = clip(tokens)
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Sampler {sampler_name} is not implemented")
        
        latents_shape =  (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        # image to image
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)

            #(Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            
            input_image_tensor = rescale(input_image_tensor, (0,255), (-1,1))
            input_image_tensor = input_image_tensor.unsqueeze(0)
            #(Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape,generator=generator, device=device)

            # run the image through the encoder of the VAE
            latents = encoder(input_image_tensor, encoder_noise)
            
            sampler.set_strength(strength = strength)
            latents =sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        
        # text to image
        else:
            # start with random noise
            latents = torch.randn(latents_shape,generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            #(1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            #(Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents 

            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise by the UNET
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # remove noise predicted by the UNET
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion) 

        decoder = models["decoder"]
        decoder.to(device)

        # (1, 4, Latents_Height, Latents_Width) -> (1, 3, Height, Width)
        image = decoder(latents)

        to_idle(decoder)

        image = rescale(image, (-1,1), (0,255), clamp = True)

        # (Batch_Size, Channel, Width, Height) -> (Batch_Size, Height, Width, Channel)
        image = image.permute(0, 2, 3, 1)
        image =  image.to("cpu", torch.uint8).numpy()
        return image[0]
    
def rescale(x, old_range, new_range, clamp = False):
    old_min, old_max = old_range
    new_min, new_max = new_range

    x = (x - old_min) 
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min

    if clamp:
        x = x.clamp(new_min, new_max)

    return x
def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange( start=0, end=160, dtype=torch.float32) / 160)

    #(1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]

    #(1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)