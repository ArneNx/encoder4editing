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

class Coach:
    def __init__(self, opts, prev_train_checkpoint=None, trigger_sync=None):
        self.opts = opts
        self.trigger_sync = trigger_sync
        self.global_step = 0

        # self.device = 'cuda:0'
        # self.opts.device = self.device
        # Initialize network
        self.net, self.device = move_to_device(pSp(self.opts))
        # self.opts.device = self.device

        # Initialize loss
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = move_to_device(LPIPS(net_type=self.opts.lpips_type))[0].eval()
        if self.opts.id_lambda > 0:
            if 'ffhq' in self.opts.dataset_type or 'celeb' in self.opts.dataset_type:
                self.id_loss = move_to_device(id_loss.IDLoss())[0].eval()
            else:
                self.id_loss = move_to_device(moco_loss.MocoLoss(opts))[0].eval()
        self.mse_loss = move_to_device(nn.MSELoss())[0].eval()
        

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()
        self.lr_scheduler = self.configure_lr_scheduler()

        # Initialize discriminator
        if self.opts.w_discriminator_lambda > 0:
            self.discriminator = move_to_device(LatentCodesDiscriminator(512, 4))[0]
            self.discriminator_optimizer = torch.optim.Adam(list(self.discriminator.parameters()),
                                                            lr=opts.w_discriminator_lr)
            self.real_w_pool = LatentCodesPool(self.opts.w_pool_size)
            self.fake_w_pool = LatentCodesPool(self.opts.w_pool_size)

        # Initialize dataset
        if "imagenet" not in opts.dataset_type:
            raise ValueError("Only ImageNet is supported for now")
        resolution = int(opts.dataset_type.split("_")[-1])
        ds_loader = FFCVDatasetLoader(dataset_cls="ImageNet", batch_size=self.opts.batch_size)
        transforms = ds_loader.get_transforms(train_data_mean= [0.485, 0.456, 0.406], train_data_std=[0.229, 0.224, 0.225], apply_augmentation=True, train_max_res=resolution, val_res=resolution)
        datasets = ds_loader.get_datasets(data_dir="/work/", orig_data_dir="/data/image_classification/ImageNet")
        data_loaders = ds_loader.get_data_loaders(datasets, transforms, seed=42, valid_size=0.01, shuffle=True, num_workers=self.opts.workers, in_memory=True)
        self.train_dataloader, self.test_dataloader = data_loaders["train"], data_loaders["test"]

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        # self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

        if prev_train_checkpoint is not None:
            self.load_from_train_checkpoint(prev_train_checkpoint)
            prev_train_checkpoint = None

        if self.opts.use_amp:
            self.scaler = GradScaler()
        self.zero_out_grad = True

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

    def train(self):
        self.start_time = time.time()
        self.net.train()
        if self.opts.progressive_steps:
            self.check_for_progressive_training_update()
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss_dict = {}
                with autocast() if self.opts.use_amp else nullcontext():
                    if self.is_training_discriminator():
                        loss_dict = self.train_discriminator(batch)
                    x, y, y_hat, latent = self.forward(batch)
                    loss, encoder_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
                    loss_dict = {**loss_dict, **encoder_loss_dict}
                    loss_dict['lr'] = self.optimizer.param_groups[0]['lr']
                if self.opts.use_amp:
                    losss = self.scaler.scale(loss) 
                if self.zero_out_grad:
                    self.optimizer.zero_grad()
                    self.zero_out_grad = False
                loss.backward()
                if self.global_step % self.opts.grad_accumulation_steps == 0:
                    self.zero_out_grad = True
                    if self.opts.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 25 == 0):
                    self.parse_and_log_images(id_logs, x, y, y_hat, title='train')
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')
                    print("Time taken for step: ", time.time() - self.start_time)
                    self.start_time = time.time()

                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1
                if self.opts.progressive_steps:
                    self.check_for_progressive_training_update()

    def check_for_progressive_training_update(self, is_resume_from_ckpt=False):
        for i in range(len(self.opts.progressive_steps)):
            if is_resume_from_ckpt and self.global_step >= self.opts.progressive_steps[i]:  # Case checkpoint
                self.net.encoder.set_progressive_stage(ProgressiveStage(i))
            if self.global_step == self.opts.progressive_steps[i]:   # Case training reached progressive step
                self.net.encoder.set_progressive_stage(ProgressiveStage(i))

    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            cur_loss_dict = {}
            if self.is_training_discriminator():
                cur_loss_dict = self.validate_discriminator(batch)
            with torch.no_grad():
                with autocast() if self.opts.use_amp else nullcontext():
                    x, y, y_hat, latent = self.forward(batch)
                    loss, cur_encoder_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
                    cur_loss_dict = {**cur_loss_dict, **cur_encoder_loss_dict}
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            if batch_idx < self.opts.image_interval:
                self.parse_and_log_images(id_logs, x, y, y_hat,
                                        title='test',
                                        subscript='{:04d}'.format(batch_idx),
                                        offset=batch_idx)

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                self.net.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    '**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def configure_optimizers(self):
        net = self.net.module if type(self.net) is nn.DataParallel else self.net
        params = list(net.encoder.parameters())
        if self.opts.train_decoder:
            params += list(net.decoder.parameters())
        else:
            self.requires_grad(net.decoder, False)
        if self.opts.optim_name == 'adam':
            optimizer = optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_lr_scheduler(self):
        lr_scheduler = None
        if self.opts.lr_scheduler == "cosine":
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.opts.max_steps-self.opts.lr_warmup,
                eta_min=0.0000001,
            )
        elif self.opts.lr_scheduler != "":
            raise ValueError("Unknown lr_scheduler: {}".format(self.opts.lr_scheduler))
        if self.opts.lr_warmup:
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0/self.opts.lr_warmup, total_iters=self.opts.lr_warmup-1 
            )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, 
            [warmup_lr_scheduler, lr_scheduler], 
            milestones=[self.opts.lr_warmup])
        return lr_scheduler

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception('{} is not a valid dataset_type'.format(self.opts.dataset_type))
        print('Loading dataset for {}'.format(self.opts.dataset_type))
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()

        train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
                                      target_root=dataset_args['train_target_root'],
                                      source_transform=transforms_dict['transform_source'],
                                      target_transform=transforms_dict['transform_gt_train'],
                                      opts=self.opts)
        test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
                                     target_root=dataset_args['test_target_root'],
                                     source_transform=transforms_dict['transform_source'],
                                     target_transform=transforms_dict['transform_test'],
                                     opts=self.opts)
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset

    def calc_loss(self, x, y, y_hat, latent):
        net = self.net.module if type(self.net) is nn.DataParallel else self.net
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.is_training_discriminator():  # Adversarial loss
            loss_disc = 0.
            dims_to_discriminate = self.get_dims_to_discriminate() if self.is_progressive_training() else \
                list(range(net.decoder.n_latent))

            for i in dims_to_discriminate:
                w = latent[:, i, :]
                fake_pred = self.discriminator(w)
                loss_disc += F.softplus(-fake_pred).mean()
            loss_disc /= len(dims_to_discriminate)
            loss_dict['encoder_discriminator_loss'] = float(loss_disc)
            loss += self.opts.w_discriminator_lambda * loss_disc

        if self.opts.progressive_steps and net.encoder.progressive_stage.value != 18 and not self.opts.encode_to_z:  # delta regularization loss
            total_delta_loss = 0
            deltas_latent_dims = net.encoder.get_deltas_starting_dimensions()

            first_w = latent[:, 0, :]
            for i in range(1, net.encoder.progressive_stage.value + 1):
                curr_dim = deltas_latent_dims[i]
                delta = latent[:, curr_dim, :] - first_w
                delta_loss = torch.norm(delta, self.opts.delta_norm, dim=1).mean()
                loss_dict[f"delta{i}_loss"] = float(delta_loss)
                total_delta_loss += delta_loss
            loss_dict['total_delta_loss'] = float(total_delta_loss)
            loss += self.opts.delta_norm_lambda * total_delta_loss

        if self.opts.id_lambda > 0:  # Similarity loss
            loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def forward(self, batch):
        x, class_id = batch
        y = x  # This should be a different view?
        x, y = x.to(self.device).float(), y.to(self.device).float()
        y_hat, latent = self.net.forward(x, class_id, return_latents=True)
        # if self.opts.dataset_type == "cars_encode":
        #     y_hat = y_hat[:, :, 32:224, :]
        return x, y, y_hat, latent

    def log_metrics(self, metrics_dict, prefix):
        wandb.log(metrics_dict, step=self.global_step)
        if self.trigger_sync is not None:
            self.trigger_sync() 

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=2, offset=0):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_image': common.log_input_image(x[i], self.opts),
                'target_image': common.tensor2im(y[i]),
                'output_image': common.tensor2im(y_hat[i]),
            }
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript, offset=offset)

    def log_images(self, name, im_data, subscript=None, log_latest=False, offset=0):
        step = self.global_step
        if log_latest:
            step = 0
        fig = common.vis_faces(im_data)
        images = wandb.Image(fig, caption=f"{step}")
        wandb.log({name: images}, step=step+offset)
        # if subscript:
        #     path = os.path.join(self.opts.exp_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
        # else:
        #     path = os.path.join(self.opts.exp_dir, name, '{:04d}.jpg'.format(step))
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        # fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        net = self.net.module if type(self.net) is nn.DataParallel else self.net
        save_dict = {
            'state_dict': net.state_dict(),
            'opts': vars(self.opts)
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = net.latent_avg

        if self.opts.save_training_data:  # Save necessary information to enable training continuation from checkpoint
            save_dict['global_step'] = self.global_step
            save_dict['optimizer'] = self.optimizer.state_dict()
            save_dict['best_val_loss'] = self.best_val_loss
            if self.opts.w_discriminator_lambda > 0:
                save_dict['discriminator_state_dict'] = self.discriminator.state_dict()
                save_dict['discriminator_optimizer_state_dict'] = self.discriminator_optimizer.state_dict()
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

        loss_dict['d_real_loss'] = float(real_loss)
        loss_dict['d_fake_loss'] = float(fake_loss)

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
        x = x.to(self.device).float()
        self.requires_grad(self.discriminator, True)

        with torch.no_grad():
            real_w, fake_w = self.sample_real_and_fake_latents(x)
        real_pred = self.discriminator(real_w)
        fake_pred = self.discriminator(fake_w)
        loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
        loss_dict['discriminator_loss'] = float(loss)

        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        # r1 regularization
        d_regularize = self.global_step % self.opts.d_reg_every == 0
        if d_regularize:
            real_w = real_w.detach()
            real_w.requires_grad = True
            real_pred = self.discriminator(real_w)
            r1_loss = self.discriminator_r1_loss(real_pred, real_w)

            self.discriminator.zero_grad()
            r1_final_loss = self.opts.r1 / 2 * r1_loss * self.opts.d_reg_every + 0 * real_pred[0]
            r1_final_loss.backward()
            self.discriminator_optimizer.step()
            loss_dict['discriminator_r1_loss'] = float(r1_final_loss)

        # Reset to previous state
        self.requires_grad(self.discriminator, False)

        return loss_dict

    def validate_discriminator(self, test_batch):
        with torch.no_grad():
            loss_dict = {}
            x, _ = test_batch
            x = x.to(self.device).float()
            real_w, fake_w = self.sample_real_and_fake_latents(x)
            real_pred = self.discriminator(real_w)
            fake_pred = self.discriminator(fake_w)
            loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
            loss_dict['discriminator_loss'] = float(loss)
            return loss_dict

    def sample_real_and_fake_latents(self, x):
        net = self.net.module if type(self.net) is nn.DataParallel else self.net
        z_dim = net.decoder.z_dim 
        c_dim = net.decoder.c_dim 
        sample_z = torch.randn(self.opts.batch_size, z_dim, device=x.device)
        sample_c_idx = torch.randint(0, c_dim, (self.opts.batch_size,), device=x.device)
        sample_c = torch.nn.functional.one_hot(sample_c_idx, num_classes=c_dim).float()
        real_w = net.decoder.get_latent(sample_z, sample_c)
        fake_w = net.encoder(x)
        if self.opts.start_from_latent_avg:
            # sample random cl
            latent_avg = net.latent_avg.to(x.device)
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
