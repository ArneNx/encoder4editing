"""
This file runs the main training/val loop
"""
import os
import json
import math
import sys
import pprint
import torch
import wandb
from wandb_osh.hooks import TriggerWandbSyncHook  # <-- New!
from argparse import Namespace

sys.path.append(".")
sys.path.append("..")

from encoder4editing.options.train_options import TrainOptions
from encoder4editing.training.coach import Coach


def main():
	opts = TrainOptions().parse()
	trigger_sync = None
	if opts.wandb_mode == "offline":
		os.environ["WANDB_MODE"] = "offline"
		trigger_sync = TriggerWandbSyncHook(communication_dir="/scratch_emmy/outputs/.wandb_commdir")  # <--- New!
	run = wandb.init(project="e4e", entity="arnenix", resume=False, dir="/scratch_emmy/outputs")
	previous_train_ckpt = None
	if opts.resume_training_from_ckpt:
		opts, previous_train_ckpt = load_train_checkpoint(opts)
		if not opts.checkpoint_path:
			opts.checkpoint_path = opts.resume_training_from_ckpt
	else:
		setup_progressive_steps(opts)
		create_initial_experiment_dir(opts)

	wandb.config.update(
		vars(opts)	
	)  
	with run:
		coach = Coach(opts, previous_train_ckpt, trigger_sync=trigger_sync)
		coach.train()

def load_train_checkpoint(opts):
	train_ckpt_path = opts.resume_training_from_ckpt
	previous_train_ckpt = torch.load(opts.resume_training_from_ckpt, map_location='cpu')
	new_opts_dict = vars(opts)
	opts = previous_train_ckpt['opts']
	opts['resume_training_from_ckpt'] = train_ckpt_path
	update_new_configs(opts, new_opts_dict)
	pprint.pprint(opts)
	opts = Namespace(**opts)
	if opts.sub_exp_dir is not None:
		sub_exp_dir = opts.sub_exp_dir
		opts.exp_dir = os.path.join(opts.exp_dir, sub_exp_dir)
		create_initial_experiment_dir(opts)
	return opts, previous_train_ckpt


def setup_progressive_steps(opts):
	log_size = int(math.log(opts.stylegan_size, 2))
	num_style_layers = 2*log_size - 2
	num_deltas = num_style_layers - 1
	if opts.progressive_start is not None:  # If progressive delta training
		opts.progressive_steps = [0]
		next_progressive_step = opts.progressive_start
		for i in range(num_deltas):
			opts.progressive_steps.append(next_progressive_step)
			next_progressive_step += opts.progressive_step_every

	assert opts.progressive_steps is None or is_valid_progressive_steps(opts, num_style_layers), \
		"Invalid progressive training input"


def is_valid_progressive_steps(opts, num_style_layers):
	return len(opts.progressive_steps) == num_style_layers and opts.progressive_steps[0] == 0


def create_initial_experiment_dir(opts):
	# if os.path.exists(opts.exp_dir):
	# 	raise Exception('Oops... {} already exists'.format(opts.exp_dir))
	os.makedirs(opts.exp_dir, exist_ok=True)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)


def update_new_configs(ckpt_opts, new_opts):
	for k, v in new_opts.items():
		if k not in ckpt_opts:
			ckpt_opts[k] = v
	if new_opts['update_param_list']:
		for param in new_opts['update_param_list']:
			ckpt_opts[param] = new_opts[param]


if __name__ == '__main__':
	main()
