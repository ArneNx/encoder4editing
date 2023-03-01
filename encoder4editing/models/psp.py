import matplotlib
from contextlib import nullcontext

matplotlib.use('Agg')
import os
import torch
from torch import nn
from encoder4editing.models.encoders import psp_encoders
# from models.stylegan2.model import Generator
from encoder4editing.configs.paths_config import model_paths
import legacy
from .generator_wrapper import GeneratorWrapper

from torch.cuda.amp import autocast

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self, opts):
        super(pSp, self).__init__()
        self.opts = opts
        # Define architecture
        # self.decoder = Generator(opts.stylegan_size, 512, 8, channel_multiplier=2)
        # Load networks.
        # network_name = "imagenet256" 
        # network_pkl = f"/pretrained_checkpoints/{network_name}.pkl"
        network_pkl = opts.stylegan_weights
        if not os.path.isfile(network_pkl):
            raise Exception(f"File {network_pkl} does not exist. Please download it from the StyleGAN-XL repo.")
        print('Loading networks from "%s"...' % network_pkl)
        device = torch.device('cuda')
        with open(network_pkl, "rb") as fp:
            generator = legacy.load_network_pkl(fp)['G_ema'].to(device) 
        self.decoder = GeneratorWrapper(generator, device)
        print(self.decoder) 
        
        self.encoder = self.set_encoder(num_ws=self.decoder.generator.num_ws)
        # self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()

    def set_encoder(self, num_ws):
        print("Using encoder:", self.opts.encoder_type)
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'Encoder4Editing':
            encoder = psp_encoders.Encoder4Editing(50, num_ws, 'ir_se', self.opts, )
        elif self.opts.encoder_type == 'SingleStyleCodeEncoder':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'SimpleLatentEncoder':
            encoder = psp_encoders.SimpleLatentEncoder(50, self.decoder.z_dim, "ir_se", self.opts, )
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        print(encoder)
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading e4e over the pSp framework from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            # self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            self.__load_latent_avg(ckpt)
        else:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(model_paths['ir_se50'])
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            # print('Loading decoder weights from pretrained!')
            # ckpt = torch.load(self.opts.stylegan_weights)
            # self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
            self.__load_latent_avg({})

    def forward(self, x, class_id, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None):
        codes = self.encoder(x)
        # if input_code:
        #     codes = x
        # else:
        #     codes = self.encoder(x)
        #     # normalize with respect to the center of an average face
        if self.opts.start_from_latent_avg:
            latent_avg = self.latent_avg[class_id]
            codes = codes + latent_avg
        #         if codes.ndim == 2:
        #             codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        #         else:
        #             codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        # if latent_mask is not None:
        #     for i in latent_mask:
        #         if inject_latent is not None:
        #             if alpha is not None:
        #                 codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
        #             else:
        #                 codes[:, i] = inject_latent[:, i]
        #         else:
        #             codes[:, i] = 0

        # input_is_latent = not input_code
        images, result_latent = self.decoder(codes,
                                             class_id,
                                             input_is_latent=self.opts.encode_to_z,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)

        # if resize:
        #     images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
        elif self.opts.start_from_latent_avg:
            # Compute mean code based on a large number of latents (10,000 here)
            with torch.no_grad():
                with autocast() if self.opts.use_amp else nullcontext():
                    self.latent_avg = self.decoder.mean_latent(10).to(self.opts.device)
        else:
            self.latent_avg = None
        if repeat is not None and self.latent_avg is not None:
            self.latent_avg = self.latent_avg.repeat(repeat, 1)
