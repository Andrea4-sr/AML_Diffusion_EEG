import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from models.custom_cond_model import Unet1D, GaussianDiffusion1D


def sample_from_model(model_path, num_samples=1, label=None, output_directory=None):

    backbone = Unet1D(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        condition=True,
        condition_classes=2,
        condition_type='attn_mlp',
    )

    model = GaussianDiffusion1D(
        backbone,
        seq_length=1000,
        timesteps=1000,
        objective='pred_x0'
    )

    model.load_state_dict(torch.load(model_path)['model'], strict=True)
    model.eval().cuda() if torch.cuda.is_available() else model.eval().cpu()

    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters: {total_params}")

    with torch.no_grad():
        if label is not None:
            label_tensor = torch.full((num_samples,), label, dtype=torch.float32).unsqueeze(1)
            label_tensor = label_tensor.cuda() if torch.cuda.is_available() else label_tensor.cpu()
            samples = model.sample(batch_size=num_samples, label=label_tensor)
        else:
            samples = model.sample(batch_size=num_samples)

    if output_directory is not None:
        os.makedirs(output_directory, exist_ok=True)
        
        existing_files = [f for f in os.listdir(output_directory) if f.startswith('sample_') and f.endswith('.npy')]
        if existing_files:
            latest_sample_index = max(int(f.split('_')[1].split('.')[0]) for f in existing_files)
        else:
            latest_sample_index = 0
        
        for i, sample in enumerate(samples):
            file_index = latest_sample_index + i + 1
            file_path = os.path.join(output_directory, f'sample_{file_index}.npy')
            np.save(file_path, sample.cpu().numpy())
    
    return samples


if __name__ == "__main__":

    model = "weights/diffusion_weights.pt"

    seizure_samples = sample_from_model(model, num_samples=5, label=1, output_directory=None)
    healthy_samples = sample_from_model(model, num_samples=5, label=0, output_directory=None)

    plt.figure(figsize=(10, 20)) 
    plt.ylim(-3, 3)
    for i in range(4):
        plt.subplot(4, 2, 2 * i + 1)
        plt.plot(healthy_samples[i].squeeze().cpu().numpy()[50:-50])
        plt.ylim(-3, 3)

        plt.subplot(4, 2, 2 * i + 2)
        plt.plot(seizure_samples[i].squeeze().cpu().numpy()[50:-50])
        plt.ylim(-3, 3)

    plt.tight_layout()
    # plt.savefig('example.png')
    plt.show()