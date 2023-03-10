# pytorch modules

import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneratorWrapper(nn.Module):
    def __init__(self, generator) -> None:
        super().__init__()
        # self.device = device
        self.generator = generator
        self.z_dim = generator.z_dim
        self.c_dim = generator.c_dim
        self.n_latent = generator.num_ws

    def mean_latent(self, n_sample):
        latent_in = torch.randn(n_sample, 1000, self.z_dim)
        class_ids = torch.arange(1000)
        class_ids = torch.nn.functional.one_hot(class_ids, num_classes=1000)
        latent_out = 0.0
        for i in range(n_sample):
            latent_out += self.generator.mapping(latent_in[i], class_ids)
        latent_out = latent_out / n_sample
        return latent_out

    def get_latent(self, z, c):
        return self.generator.mapping(z, c)

    def forward(
            self,
            style_codes,
            class_ids,
            return_latents=False,
            return_features=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            noise=None,
            randomize_noise=True,
    ):
        # if input_is_latent:
        #     raise NotImplementedError(
        #         "input_is_latent is not supported in GeneratorWrapper"
        #     )
        # styles = G_original.mapping(z_samples, c_samples, truncation_psi=0.5)

        # territory_indicator_ws = [get_morphed_w_code(styles.unsqueeze(0), w_batch) for w_code in w_samples]

        if input_is_latent:
            class_ids = torch.nn.functional.one_hot(class_ids, num_classes=1000)
            style_codes = self.generator.mapping(style_codes, class_ids)

        image = self.generator.synthesis(style_codes, noise_mode='none', force_fp32=False)


        # if return_latents:
        #     return image, styles
        # else:
        return image, style_codes
