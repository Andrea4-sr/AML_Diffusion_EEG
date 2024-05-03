import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as pl
from output_shape import output_shape
from sklearn.model_selection import train_test_split
import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle

torch.set_float32_matmul_precision('medium')

from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Dataset1D, Trainer1D
from denoising_diffusion_pytorch.karras_unet_1d import KarrasUnet1D


if __name__ == '__main__':

    filenames = [f for f in os.listdir('../data/segments/') if os.path.isfile(os.path.join('../data/labels/', f))]

    data = []
    labels = []

    for filename in tqdm.tqdm(filenames):
        segment = np.load(os.path.join('../data/segments/', filename))
        label = np.load(os.path.join('../data/labels/', filename))

        segment = segment[:, 0, :1000]

        data.append(segment.astype(np.float16))
        labels.append(label.astype(np.float16))
        
    data = np.concatenate(data, axis=0).astype(np.float16)
    labels = np.concatenate(labels, axis=0).astype(np.float16)

    normalized_data = (data - data.min()) / (data.max() - data.min())
    dataset = Dataset1D(torch.from_numpy(normalized_data).unsqueeze(1))


    backbone = Unet1D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 1
    )

    model = GaussianDiffusion1D(
        backbone,
        seq_length = 1000,
        timesteps = 1000,
        objective = 'pred_v'
    )


    trainer = Trainer1D(
        model,
        dataset = dataset,
        train_batch_size = 64,
        train_lr = 8e-5,
        train_num_steps = 7000,           # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
    )

    trainer.train()