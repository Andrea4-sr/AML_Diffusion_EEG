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

from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from denoising_diffusion_pytorch.karras_unet_1d import KarrasUnet1D


class EEGLoader(pl.LightningDataModule):
    def __init__(self, data_dir='data', batch_size=128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        segments_path = os.path.join(self.data_dir, '..data/segments/')
        labels_path = os.path.join(self.data_dir, '..data/labels/')
        filenames = [f for f in os.listdir(segments_path) if os.path.isfile(os.path.join(labels_path, f))]

        data = []
        labels = []

        for filename in tqdm.tqdm(filenames):
            segment = np.load(os.path.join(segments_path, filename))
            label = np.load(os.path.join(labels_path, filename))

            segment = segment[:, 0, :1000]

            data.append(segment.astype(np.float16))
            labels.append(label.astype(np.float16))
            
        data = np.concatenate(data, axis=0).astype(np.float16)
        labels = np.concatenate(labels, axis=0).astype(np.float16)

        self.train_data, self.val_data, self.train_labels, self.val_labels = train_test_split(data, labels, test_size=0.2, random_state=3)

        self.train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.train_data), torch.from_numpy(self.train_labels))
        self.val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.val_data), torch.from_numpy(self.val_labels))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True)


class _1D_Diffusion(pl.LightningModule):
    def __init__(self, debug = False, learning_rate=1e-4):
        super().__init__()
        self.debug = debug
        self.learning_rate = learning_rate

        self.backbone = Unet1D(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            channels = 1
        )

        self.model = GaussianDiffusion1D(
            self.backbone,
            seq_length = 1000,
            timesteps = 1000,
            objective = 'pred_v'
        )

    @output_shape
    def forward(self, x):
        loss = self.model(x.unsqueeze(1))

        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch

        loss = self(x)

        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch

        loss = self(x)

        self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return loss

    def on_validation_epoch_end(self):

        plt.plot(self.model.sample(batch_size=1).detach().cpu().squeeze())
        plt.savefig(f"{self.trainer.global_step}")
        plt.close()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.999), amsgrad=True)
        return optimizer
    
# if __name__ == '__main__':
#     _1D_Diffusion(debug=True)(torch.rand(32, 1, 1000))

# trainer = pl.Trainer( 
#                 devices=1,
#                 max_steps=10000,
#                 enable_model_summary=False,
#                 enable_progress_bar=True,
#                 gradient_clip_val=1.5,
#                 precision='16-mixed')

# trainer.fit(_1D_Diffusion(), EEGLoader())

# -------------------------------------------------------------

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

print(data.shape)
print(labels.shape)
exit()

trainer = Trainer1D(
    _1D_Diffusion(),
    dataset = EEGLoader(),
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
)
trainer.train()