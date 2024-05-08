import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, einsum, Tensor
import torch.nn.init as init
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from accelerate import Accelerator
from ema_pytorch import EMA

from tqdm.auto import tqdm

from models.custom_cond_model import GaussianDiffusion1D

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def standarize(t):
    return (t - t.mean()) / t.std()

# data

# class Dataset1D(Dataset):
#     def __init__(self, tensor: Tensor):
#         super().__init__()
#         self.tensor = tensor.clone()

#     def __len__(self):
#         return len(self.tensor)

#     def __getitem__(self, idx):
#         return self.tensor[idx].clone()

def load_data_and_labels(base_path, class_labels):
    data, labels = [], []
    for class_name, label in class_labels.items():
        class_path = os.path.join(base_path, class_name)
        for file_name in tqdm(os.listdir(class_path)):
            if file_name.endswith('.npy'):
                file_path = os.path.join(class_path, file_name)
                sample = np.load(file_path)
                data.append(sample.astype(np.float32))
                labels.append(np.float32(label))
    return data, labels

class Dataset1D(Dataset):
    def __init__(self, base_path, class_labels, no_classes = False, test = False):
        self.no_classes = no_classes
        if test:
            self.data = np.random.randn(99739, 1000)
            self.labels = np.random.randint(0, 3, (99739, 1))
        else:
            self.data, self.labels = load_data_and_labels(base_path, class_labels)
            self.data, self.labels = np.vstack(self.data), np.vstack(self.labels)
            self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())

        self.data = torch.from_numpy(self.data).unsqueeze(1).to(torch.float32)
        self.labels = torch.from_numpy(self.labels).to(torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.no_classes:
            return self.data[idx]
        else:
            return self.data[idx], self.labels[idx]

class Trainer1D(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion1D,
        dataset: Dataset,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1.,
        conditional = False,
        wandb = None,
    ):
        super().__init__()

        # wandb

        self.conditional = conditional

        # wandb

        self.wandb = wandb

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        # dataset and dataloader

        dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 8, persistent_workers=True)
        
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        os.makedirs(results_folder, exist_ok = True)
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    if self.conditional:
                        data, labels = next(self.dl)
                        data, labels = data.to(device), labels.to(device)
                    else:
                        data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                        if exists(self.wandb) and self.step % 20 == 0: self.wandb.log({'loss': loss})
                        if exists(self.wandb) and self.step % 20 == 0: self.wandb.log({'logloss': torch.log(loss)})

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    # if self.step == 100:
                    #     self.ema.ema_model.eval()

                    #     with torch.no_grad():
                    #         batches = num_to_groups(self.num_samples, self.batch_size)
                    #         all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=1, label=torch.tensor([0], dtype=torch.float32)), batches))

                    #     plt.figure(figsize=(10, 5))
                    #     plt.plot(all_samples_list[0].squeeze().cpu().numpy())
                    #     plt.savefig(str(self.results_folder / f'{self.step}_sample_0.png'))
                    #     plt.close()
                    #     if exists(self.wandb): self.wandb.log({"start_pred": [self.wandb.Image(os.path.join(self.results_folder, f'{self.step}_sample_0.png'), caption="Start Prediction 0")]})

                    #     with torch.no_grad():
                    #         batches = num_to_groups(self.num_samples, self.batch_size)
                    #         all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=1, label=torch.tensor([1], dtype=torch.float32)), batches))

                    #     plt.figure(figsize=(10, 5))
                    #     plt.plot(all_samples_list[0].squeeze().cpu().numpy())
                    #     plt.savefig(str(self.results_folder / f'{self.step}_sample_1.png'))
                    #     plt.close()
                    #     if exists(self.wandb): self.wandb.log({"start_pred": [self.wandb.Image(os.path.join(self.results_folder, f'{self.step}_sample_1.png'), caption="Start Prediction 1")]})

                    #     with torch.no_grad():
                    #         batches = num_to_groups(self.num_samples, self.batch_size)
                    #         all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=1, label=torch.tensor([2], dtype=torch.float32)), batches))

                    #     plt.figure(figsize=(10, 5))
                    #     plt.plot(all_samples_list[0].squeeze().cpu().numpy())
                    #     plt.savefig(str(self.results_folder / f'{self.step}_sample_2.png'))
                    #     plt.close()
                    #     if exists(self.wandb): self.wandb.log({"start_pred": [self.wandb.Image(os.path.join(self.results_folder, f'{self.step}_sample_2.png'), caption="Start Prediction 2")]})

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        if self.conditional:
                            with torch.no_grad():
                                milestone = self.step // self.save_and_sample_every
                                batches = num_to_groups(self.num_samples, self.batch_size)
                                all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=1, label=torch.tensor([0], dtype=torch.float32)), batches))

                            plt.figure(figsize=(10, 5))
                            plt.plot(all_samples_list[0].squeeze().cpu().numpy())
                            plt.savefig(str(self.results_folder / f'{self.step}_sample_0.png'))
                            plt.close()
                            if exists(self.wandb): self.wandb.log({f"pred 0": [self.wandb.Image(os.path.join(self.results_folder, f'{self.step}_sample_0.png'), caption=f"Prediction 0 step {self.step}")]})

                            with torch.no_grad():
                                milestone = self.step // self.save_and_sample_every
                                batches = num_to_groups(self.num_samples, self.batch_size)
                                all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=1, label=torch.tensor([1], dtype=torch.float32)), batches))

                            plt.figure(figsize=(10, 5))
                            plt.plot(all_samples_list[0].squeeze().cpu().numpy())
                            plt.savefig(str(self.results_folder /  f'{self.step}_sample_1.png'))
                            plt.close()
                            if exists(self.wandb): self.wandb.log({f"pred 1": [self.wandb.Image(os.path.join(self.results_folder, f'{self.step}_sample_1.png'), caption=f"Prediction 1 step {self.step}")]})

                            with torch.no_grad():
                                milestone = self.step // self.save_and_sample_every
                                batches = num_to_groups(self.num_samples, self.batch_size)
                                all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=1, label=torch.tensor([2], dtype=torch.float32)), batches))

                            plt.figure(figsize=(10, 5))
                            plt.plot(all_samples_list[0].squeeze().cpu().numpy())
                            plt.savefig(str(self.results_folder / f'{self.step}_sample_2.png'))
                            plt.close()
                            if exists(self.wandb): self.wandb.log({f"pred 2": [self.wandb.Image(os.path.join(self.results_folder, f'{self.step}_sample_2.png'), caption=f"Prediction 2 step {self.step}")]})

                        else:
                            with torch.no_grad():
                                milestone = self.step // self.save_and_sample_every
                                batches = num_to_groups(self.num_samples, self.batch_size)
                                all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=1), batches))

                            plt.figure(figsize=(10, 5))
                            plt.plot(all_samples_list[0].squeeze().cpu().numpy())
                            plt.savefig(str(self.results_folder / f'{self.step}_sample.png'))
                            plt.close()
                            if exists(self.wandb): self.wandb.log({f"pred": [self.wandb.Image(os.path.join(self.results_folder, f'{self.step}_sample.png'), caption=f"Prediction step {self.step}")]})

                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')