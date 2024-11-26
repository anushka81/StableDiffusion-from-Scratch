import torch
import numpy as np


class DDPMSampler:

    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps=1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
    ):
        self.betas = (
            torch.linspace(
                beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32
            )
            ** 2
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timestep(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        # 999 998 997 ... 2 1 0
        # 999, 999 -20, 999 - 40, 999 - 60, 999 - 80, 999 - 100 = 50 steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)

    def get_prev_timestep(self, timestep: int) -> int:
        prev_t = timestep - (self.num_training_steps // self.num_inference_steps)
        return prev_t

    def get_variance(self, timestep: int) -> torch.Tensor:
        prev_t = self.get_prev_timestep(timestep)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # computed using formula (6) (7) of ddpm paper
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)

        return variance

    def set_strength(self, strength=1):
        """
        Set how much noise to add to the input image.
        More noise (strength ~ 1) means that the output will be further from the input image.
        Less noise (strength ~ 0) means that the output will be closer to the input image.
        """
        # start_step is the number of noise levels to skip
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        prev_t = self.get_prev_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alph_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alph_t

        # compute the predicted original sample using formula (15) of ddpm paper
        pred_orig_sample = (
            latents - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5

        # compute the coefficients for pred_original_sample and current sample x_t
        pred_orig_sample_coeff = (alpha_prod_t_prev**0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = (current_alph_t**0.5 * beta_prod_t_prev) / beta_prod_t

        # predict the sample using formula (7) of ddpm paper
        pred_prev_sample = (
            pred_orig_sample_coeff * pred_orig_sample + current_sample_coeff * latents
        )

        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=device,
                dtype=model_output.dtype,
            )
            variance = (self.get_variance(t) ** 0.5) * noise

        # N(0,1) -> N(mu, sigma^2)
        # X = mu + sigma * Z where Z ~ N(0,1)
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    # for image to image noise is applied on latent
    def add_noise(
        self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor
    ) -> torch.FloatTensor:
        alpha_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (
            1 - alpha_cumprod[timesteps]
        ) ** 0.5  # standard deviation
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # According to DDPM paper
        # equation (4)
        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype,
        )
        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples