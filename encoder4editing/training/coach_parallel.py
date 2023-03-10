import os
import random
import time
from encoder4editing.training.trainer import Trainer
import matplotlib
import matplotlib.pyplot as plt
from contextlib import nullcontext
import wandb
import numpy as np
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

        self.trainer = Trainer(self.opts, prev_train_checkpoint)
        # self.opts.device = self.device

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()
        self.lr_scheduler = self.configure_lr_scheduler()

        self.trainer, self.device = move_to_device(self.trainer)
        # Initialize discriminator
        if self.opts.w_discriminator_lambda > 0:
            disc_params = self.trainer.discriminator.parameters() if not isinstance(self.trainer, nn.DataParallel) else self.trainer.module.discriminator.parameters()
            self.discriminator_optimizer = torch.optim.Adam(disc_params,
                                                            lr=opts.w_discriminator_lr)

        # Initialize dataset
        if "imagenet" not in opts.dataset_type:
            raise ValueError("Only ImageNet is supported for now")
        resolution = int(opts.dataset_type.split("_")[-1])
        ds_loader = FFCVDatasetLoader(dataset_cls="ImageNet", batch_size=self.opts.batch_size)
        transforms = ds_loader.get_transforms(train_data_mean= [0.485, 0.456, 0.406], train_data_std=[0.229, 0.224, 0.225], apply_augmentation=True, train_max_res=resolution, val_res=resolution, dtype=np.float32)
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

        if self.opts.use_amp:
            self.scaler = GradScaler()
        self.zero_out_grad = True

    def optimize(self, loss, optimizer):
        if self.opts.use_amp:
            loss = self.scaler.scale(loss) 
        if self.zero_out_grad:
            optimizer.zero_grad()
        loss.backward()
        if self.trainer.global_step % self.opts.grad_accumulation_steps == 0:
            if self.opts.use_amp:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()

    
    def train(self):
        torch.backends.cudnn.enabled = False
        self.start_time = time.time()
        trainer = self.trainer if not isinstance(self.trainer, nn.DataParallel) else self.trainer.module
        trainer.train()
        if self.opts.progressive_steps:
            self.check_for_progressive_training_update()
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss_dict = {}
                with autocast() if self.opts.use_amp else nullcontext():
                    loss, disc_loss, r1_disc_loss, x, y, y_hat = self.trainer.forward(batch, training=True)
                print("LOSS", loss.shape)
                loss_dict = {
                    "loss": float(loss),
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "discriminator_loss": float(disc_loss) if disc_loss is not None else None,
                    "discriminator_r1_loss": float(r1_disc_loss) if r1_disc_loss is not None else None,
                }
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimize(loss, self.optimizer)

                discriminator = trainer.discriminator 
                trainer.requires_grad(discriminator, True) 
                if r1_disc_loss is not None:
                    disc_loss = disc_loss + r1_disc_loss
                self.optimize(disc_loss, self.discriminator_optimizer) if disc_loss is not None else None
                # self.optimize(r1_disc_loss, self.discriminator_optimizer) if disc_loss is not None else None
                trainer.requires_grad(discriminator, False) 

                self.zero_out_grad = self.global_step % self.opts.grad_accumulation_steps == 0  # for first iteration of accumulation

                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 25 == 0):
                    self.parse_and_log_images(x, y, y_hat, title='train')
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

                trainer.global_step += 1
                if self.opts.progressive_steps:
                    trainer.check_for_progressive_training_update()

    def validate(self):
        # trainer.eval() if not isinstance(self.trainer, nn.DataParallel) else trainer.module.eval()
        self.trainer.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            cur_loss_dict = {}
            with torch.no_grad():
                with autocast() if self.opts.use_amp else nullcontext():
                    loss, disc_loss, r1_disc_loss, x, y, y_hat = self.trainer.forward(batch, training=False)  
                    cur_loss_dict = {
                        "loss": float(loss),
                        "discriminator_loss": float(disc_loss) if disc_loss is not None else None,
                        "discriminator_r1_loss": float(r1_disc_loss) if r1_disc_loss is not None else None,
                    }
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            if batch_idx < self.opts.image_interval:
                self.parse_and_log_images(x, y, y_hat,
                                        title='test',
                                        subscript='{:04d}'.format(batch_idx),
                                        offset=batch_idx)

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                self.trainer.train() #if not isinstance(self.trainer, nn.DataParallel) else self.trainer.module.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.trainer.train() # if not isinstance(self.trainer, nn.DataParallel) else self.trainer.module.train()
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
        params = self.trainer.get_net_parameters() 
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

    @property
    def global_step(self):
        if isinstance(self.trainer, nn.DataParallel):
            return self.trainer.module.global_step
        return self.trainer.global_step

    def log_metrics(self, metrics_dict, prefix):
        wandb.log(metrics_dict, step=self.global_step)
        if self.trigger_sync is not None:
            self.trigger_sync() 

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, x, y, y_hat, title, subscript=None, display_count=2, offset=0):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_image': common.log_input_image(x[i], self.opts),
                'target_image': common.tensor2im(y[i]),
                'output_image': common.tensor2im(y_hat[i]),
            }
            # if id_logs is not None:
            #     for key in id_logs[i]:
            #         cur_im_data[key] = id_logs[i][key]
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
        save_dict = self.trainer.get_save_dict()

        if self.opts.save_training_data:  # Save necessary information to enable training continuation from checkpoint
            save_dict['optimizer'] = self.optimizer.state_dict()
            if self.opts.w_discriminator_lambda > 0:
                save_dict['discriminator_optimizer_state_dict'] = self.discriminator_optimizer.state_dict()
        return save_dict

