import os
import random
import time
import matplotlib
import matplotlib.pyplot as plt
from contextlib import nullcontext
import wandb
matplotlib.use('Agg')

import torch
from torch import nn, autograd, optim
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from encoder4editing.utils import common, train_utils
from encoder4editing.criteria import id_loss, moco_loss
from encoder4editing.configs import data_configs
from encoder4editing.datasets.images_dataset import ImagesDataset
from encoder4editing.criteria.lpips.lpips import LPIPS
from encoder4editing.models.psp import pSp
from encoder4editing.models.latent_codes_pool import LatentCodesPool
from encoder4editing.models.discriminator import LatentCodesDiscriminator
from encoder4editing.models.encoders.psp_encoders import ProgressiveStage
from encoder4editing.training.ranger import Ranger

from xaug.dataset.ffcv_imagenet_loader import FFCVDatasetLoader

random.seed(0)
torch.manual_seed(0)


def move_to_device(model, gpu=True, multi_gpu=True, channels_last=False):
    """
    Moves given model to GPU(s) if they are available
    :param model: (torch.nn.Module) model to move
    :param gpu: (bool) if True attempt to move to GPU
    :param multi_gpu: (bool) if True attempt to use multi-GPU
    :return: torch.nn.Module, str
    """
    device = "cuda" if torch.cuda.is_available() and gpu else "cpu"
    if multi_gpu and torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    if channels_last:
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    return model, device

class Trainer(nn.Module):

    def __init__(self, opts, prev_train_checkpoint=None):
        self.opts = opts
        self.global_step = 0

        super().__init__()
        self.net = pSp(self.opts)


        # Initialize loss
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type=self.opts.lpips_type)
        if self.opts.id_lambda > 0:
            if 'ffhq' in self.opts.dataset_type or 'celeb' in self.opts.dataset_type:
                self.id_loss = id_loss.IDLoss()
            else:
                self.id_loss = moco_loss.MocoLoss(opts)
        self.mse_loss = nn.MSELoss()
        
        # Initialize discriminator
        if self.opts.w_discriminator_lambda > 0:
            self.discriminator = LatentCodesDiscriminator(512, 4)
            # self.discriminator_optimizer = torch.optim.Adam(list(self.discriminator.parameters()),
            #                                                 lr=opts.w_discriminator_lr)
            self.real_w_pool = LatentCodesPool(self.opts.w_pool_size)
            self.fake_w_pool = LatentCodesPool(self.opts.w_pool_size)

        if prev_train_checkpoint is not None:
            self.load_from_train_checkpoint(prev_train_checkpoint)
            prev_train_checkpoint = None

    def load_from_train_checkpoint(self, ckpt):
        print('Loading previous training data...')
        self.global_step = ckpt['global_step'] + 1
        self.best_val_loss = ckpt['best_val_loss']
        self.net.load_state_dict(ckpt['state_dict'])

        if self.opts.keep_optimizer:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.opts.w_discriminator_lambda > 0:
            self.discriminator.load_state_dict(ckpt['discriminator_state_dict'])
            self.discriminator_optimizer.load_state_dict(ckpt['discriminator_optimizer_state_dict'])
        if self.opts.progressive_steps:
            self.check_for_progressive_training_update(is_resume_from_ckpt=True)
        print(f'Resuming training from step {self.global_step}')

    def forward(self, batch, training=True):
        disc_loss = r1_disc_loss = None
        if self.is_training_discriminator():
            if training:
                loss_dict, disc_loss, r1_disc_loss = self.train_discriminator(batch)
            else:
                loss_dict = self.validate_discriminator(batch)
        x, y, y_hat, latent = self.net_forward(batch)
        loss, encoder_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
        # loss_dict = {**loss_dict, **encoder_loss_dict}
        return loss, disc_loss, r1_disc_loss, x, y, y_hat

    def check_for_progressive_training_update(self, is_resume_from_ckpt=False):
        for i in range(len(self.opts.progressive_steps)):
            if is_resume_from_ckpt and self.global_step >= self.opts.progressive_steps[i]:  # Case checkpoint
                self.net.encoder.set_progressive_stage(ProgressiveStage(i))
            if self.global_step == self.opts.progressive_steps[i]:   # Case training reached progressive step
                self.net.encoder.set_progressive_stage(ProgressiveStage(i))

    def calc_loss(self, x, y, y_hat, latent):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.is_training_discriminator():  # Adversarial loss
            loss_disc = 0.
            dims_to_discriminate = self.get_dims_to_discriminate() if self.is_progressive_training() else \
                list(range(self.net.decoder.n_latent))

            for i in dims_to_discriminate:
                w = latent[:, i, :]
                fake_pred = self.discriminator(w)
                loss_disc += F.softplus(-fake_pred).mean()
            loss_disc /= len(dims_to_discriminate)
            # loss_dict['encoder_discriminator_loss'] = float(loss_disc)
            loss += self.opts.w_discriminator_lambda * loss_disc

        if self.opts.progressive_steps and self.net.encoder.progressive_stage.value != 18 and not self.opts.encode_to_z:  # delta regularization loss
            total_delta_loss = 0
            deltas_latent_dims = self.net.encoder.get_deltas_starting_dimensions()

            first_w = latent[:, 0, :]
            for i in range(1, self.net.encoder.progressive_stage.value + 1):
                curr_dim = deltas_latent_dims[i]
                delta = latent[:, curr_dim, :] - first_w
                delta_loss = torch.norm(delta, self.opts.delta_norm, dim=1).mean()
                # loss_dict[f"delta{i}_loss"] = float(delta_loss)
                total_delta_loss += delta_loss
            # loss_dict['total_delta_loss'] = float(total_delta_loss)
            loss += self.opts.delta_norm_lambda * total_delta_loss

        if self.opts.id_lambda > 0:  # Similarity loss
            loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
            # loss_dict['loss_id'] = float(loss_id)
            # loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            # loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y)
            # loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        # loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def net_forward(self, batch):
        x, class_id = batch
        y = x  # This should be a different view?
        # x, y = x.to(self.device).float(), y.to(self.device).float()
        y_hat, latent = self.net.forward(x, class_id, return_latents=True)
        # if self.opts.dataset_type == "cars_encode":
        #     y_hat = y_hat[:, :, 32:224, :]
        return x, y, y_hat, latent

    def get_net_parameters(self):
        params = list(self.net.encoder.parameters())
        if self.opts.train_decoder:
            params += list(self.net.decoder.parameters())
        else:
            self.requires_grad(self.net.decoder, False)
        return params

    def get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts)
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = self.net.latent_avg

        if self.opts.save_training_data:  # Save necessary information to enable training continuation from checkpoint
            save_dict['global_step'] = self.global_step
            save_dict['best_val_loss'] = self.best_val_loss
            if self.opts.w_discriminator_lambda > 0:
                save_dict['discriminator_state_dict'] = self.discriminator.state_dict()
        return save_dict

    def get_dims_to_discriminate(self):
        net = self.net.module if type(self.net) is nn.DataParallel else self.net
        deltas_starting_dimensions = net.encoder.get_deltas_starting_dimensions()
        return deltas_starting_dimensions[:net.encoder.progressive_stage.value + 1]

    def is_progressive_training(self):
        return self.opts.progressive_steps is not None

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Discriminator ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def is_training_discriminator(self):
        return self.opts.w_discriminator_lambda > 0 and not self.opts.encode_to_z

    @staticmethod
    def discriminator_loss(real_pred, fake_pred, loss_dict):
        real_loss = F.softplus(-real_pred).mean()
        fake_loss = F.softplus(fake_pred).mean()

        # loss_dict['d_real_loss'] = float(real_loss)
        # loss_dict['d_fake_loss'] = float(fake_loss)

        return real_loss + fake_loss

    @staticmethod
    def discriminator_r1_loss(real_pred, real_w):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_w, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def train_discriminator(self, batch):  # Trains discriminator network to distinguish fake and real latents
        loss_dict = {}
        x, _ = batch
        # x = x.to(self.device).float()
        self.requires_grad(self.discriminator, True)

        with torch.no_grad():
            real_w, fake_w = self.sample_real_and_fake_latents(x)
        real_pred = self.discriminator(real_w)
        fake_pred = self.discriminator(fake_w)
        loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
        # loss_dict['discriminator_loss'] = float(loss)

        # self.discriminator_optimizer.zero_grad()
        # loss.backward()
        # self.discriminator_optimizer.step()

        # r1 regularization
        d_regularize = self.global_step % self.opts.d_reg_every == 0
        if d_regularize:
            real_w = real_w.detach()
            real_w.requires_grad = True
            real_pred = self.discriminator(real_w)
            r1_loss = self.discriminator_r1_loss(real_pred, real_w)

            r1_final_loss = self.opts.r1 / 2 * r1_loss * self.opts.d_reg_every + 0 * real_pred[0]
            # self.discriminator.zero_grad()
            # r1_final_loss.backward()
            # self.discriminator_optimizer.step()
            # loss_dict['discriminator_r1_loss'] = r1_final_loss.detach()
            return loss_dict,loss, r1_final_loss 

        # Reset to previous state
        self.requires_grad(self.discriminator, False)

        return loss_dict,loss, None 



    def validate_discriminator(self, test_batch):
        with torch.no_grad():
            loss_dict = {}
            x, _ = test_batch
            real_w, fake_w = self.sample_real_and_fake_latents(x)
            real_pred = self.discriminator(real_w)
            fake_pred = self.discriminator(fake_w)
            loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
            loss_dict['discriminator_loss'] = float(loss)
            return loss_dict

    def sample_real_and_fake_latents(self, x):
        z_dim = self.net.decoder.z_dim 
        c_dim = self.net.decoder.c_dim 
        sample_z = torch.randn(x.shape[0], z_dim, device=x.device)
        sample_c_idx = torch.randint(0, c_dim, (x.shape[0],), device=x.device)
        sample_c = torch.nn.functional.one_hot(sample_c_idx, num_classes=c_dim).float()
        real_w = self.net.decoder.get_latent(sample_z, sample_c)
        fake_w = self.net.encoder(x)
        if self.opts.start_from_latent_avg:
            # sample random cl
            latent_avg = self.net.latent_avg.to(x.device)
            sampled_avg = latent_avg[sample_c_idx]
            fake_w = fake_w + sampled_avg #self.net.latent_avg[sample_c_idx]
        if self.is_progressive_training():  # When progressive training, feed only unique w's
            dims_to_discriminate = self.get_dims_to_discriminate()
            fake_w = fake_w[:, dims_to_discriminate, :]
        if self.opts.use_w_pool:
            real_w = self.real_w_pool.query(real_w)
            fake_w = self.fake_w_pool.query(fake_w)
        if fake_w.ndim == 3:
            fake_w = fake_w[:, 0, :]
        return real_w, fake_w
