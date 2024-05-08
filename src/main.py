import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import lightning as pl
from output_shape import output_shape
from sklearn.model_selection import train_test_split
import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import inspect
import wandb

torch.set_float32_matmul_precision('medium')

from models.non_cond_model import Unet1D, GaussianDiffusion1D
from models.custom_cond_model import Unet1D as custom_Unet1D, GaussianDiffusion1D as custom_GaussianDiffusion1D
# from models.karras_cond_model import Unet1D as karras_Unet1D, GaussianDiffusion1D as karras_GaussianDiffusion1D
from trainer import Trainer1D, Dataset1D

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Unet1D', help="What model to use for training")
    parser.add_argument('--dim', type=int, default=64, help="Dimension of the model")
    parser.add_argument('--condition', action='store_true', default=False, help="Use conditioning for the model")
    parser.add_argument('--condition_classes', type=int, default=3, help="Number of classes for conditioning")
    parser.add_argument('--condition_type', type=str, default='input_concat', help="Type of conditioning to use")
    parser.add_argument('--channels', type=int, default=1, help="Number of channels in the model")

    parser.add_argument('--data_path', type=str, default='data/train_250hz_05_70_n60_CZ', help="Path to the data")
    parser.add_argument('--objective', type=str, default='pred_v', help="Objective to use for training: 'pred_noise', 'pred_x0', 'pred_v'")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--train_steps', type=int, default=10000, help="Number of training steps")
    parser.add_argument('--wandb', action='store_true', default=False, help="Use wandb for logging")
    parser.add_argument('--name', type=str, default='test', help="Name of the run")
    args = parser.parse_args()

    model_args = {arg: value for arg, value in vars(args).items() if value is not None and arg in inspect.signature(eval(args.model).__init__).parameters}
    model = eval(args.model)(**model_args)

    class_labels = {
        'non_seizure': 0.0,
        'fnsz': 1.0,
        'gnsz': 2.0
    }

    dataset = Dataset1D(os.path.join('data', 'train_250hz_05_70_n60_CZ'), class_labels, no_classes=True if not args.condition else False)

    model_wrapper = GaussianDiffusion1D(
        model,
        seq_length = 1000,
        timesteps = 1000,
        objective = args.objective,
    )

    if args.wandb:
        wandb.init(project="AML-Diffusion", entity="avocardio", name=args.name)
        wandb.config.update(args)
        wandb.run.name = args.name

    trainer = Trainer1D(
        model_wrapper,
        dataset = dataset,
        train_batch_size = args.batch_size,
        train_lr = 8e-5,
        train_num_steps = args.train_steps,          
        gradient_accumulate_every = 2,
        ema_decay = 0.995,          
        amp = True,            
        results_folder = os.path.join('results', args.name),
        wandb = wandb if args.wandb else None,
    )

    trainer.train()