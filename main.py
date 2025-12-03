#!/usr/bin/env python3
import os
import torch
import wandb
import logging
import argparse
from datasets.dataset import get_dataloaders
from models.model import BindModel

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch import loggers as pl_loggers

modality_space = [
    'vision', # Vi
    'text', # Te
    'audio', # Au
    'depth', # De
    'thermal', # Th
    'imu', # Im
    'tactile', # Ta
    'point', # Po
]


def parse_args():
    parser = argparse.ArgumentParser(description="Bind")
    parser.add_argument("-i", "--info", default = "", type=str)
    parser.add_argument("--checkpoints", type=str, default='./checkpoints/', help="")
    parser.add_argument("--log_path", default = "./log/", type=str)
    parser.add_argument("--data_path", default = "./datasets/", type=str)
    parser.add_argument("--seed", default = 0, type=int)
    parser.add_argument("--gpus", default = 1, type=int)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--oom", default='vision', type=str)
    parser.add_argument("--modalities", default=['audio', 'vision', 'text'])

    # Optimizer setting
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--momentum_betas", nargs=2, type=float, default=[0.9, 0.95],
                        help="Momentum beta 1 and 2 for Adam optimizer")
    parser.add_argument("--max_epoch", default = 50, type=int)
    parser.add_argument("--cur_epoch", default = 0, type=int)
    parser.add_argument("--warmup_epoch", default = 5, type=int)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")

    # Model setting
    parser.add_argument("--model_type", default = 'imagebind', type=str)
    parser.add_argument("--num_classes", default = 7, type=int)
    parser.add_argument("--lora", action="store_true", help="Use LoRA")
    parser.add_argument("--lora_rank", default = 4, type=int, help="LoRA rank")
    parser.add_argument("--linear_probing", action="store_true", help="Use linear probing")
    parser.add_argument("--save_models", action="store_true", help="Save models")
    parser.add_argument("--out_embed_dim", default = 1024, type=int)

    # Dataset setting
    parser.add_argument("--datasets", default = "msrvtt", nargs='+', type=str)
    parser.add_argument("--num_classes", default = 20, type=int)
    parser.add_argument("--num_workers", default = 4, type=int)
    parser.add_argument("--batch_size", default = 128, type=int)
    parser.add_argument("--rate", type=float, default=0.1, help="Sampling rate")

    args = parser.parse_args()
    return args


def main():
    wandb.login()
    args = parse_args()
    os.makedirs(args.log_path, exist_ok=True)

    # Set experiment properties
    seed_everything(args.seed, workers=True)
    torch.backends.cudnn.determinstic = True
    torch.set_float32_matmul_precision('medium')

    loggers = []
    wandb.init(project=args.info, config=args)
    logger = pl_loggers.WandbLogger(
        save_dir=args.log_path,
        name=args.info)
    loggers.append(logger)
    for key, value in vars(args).items():
        logging.info(f"\t\t{key}\t\t\t{value}")

    bindmodel = BindModel(args, args.model_type, args.modalities)
    dataloaders = get_dataloaders(args, args.datasets, args.model_type, args.modalities, bindmodel.models)

    trainer = Trainer(accelerator="gpu" if "cuda" in args.device else "cpu", 
                      devices=1 if ":" not in args.device else [int(args.device.split(":")[1])],
                      deterministic=True, max_epochs=args.max_epoch, gradient_clip_val=args.gradient_clip_val,
                      logger=loggers)
    trainer.fit(bindmodel, dataloaders['train'], dataloaders['test'])


if __name__ == "__main__":

    logging.info('starting...')

    main()

