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

        for i, sample in enumerate(samples):
            file_path = os.path.join(output_directory, f'sample_{i+1}.npy')
            np.save(file_path, sample.cpu().numpy())
    
    return samples


if __name__ == "__main__":


    # model = "results\++ long 2 class Unet1D attn mlp - pred_x0\model-50.pt"  # ************ Weird depressions
    # model = "results\+ comb long 2 class Unet1D attn mlp - pred_x0\model-50.pt" # ************ Too noisy healthy data
    model = "results\++ 2 Unet1D attn mlp - pred_x0\model-15.pt" # ************ This one is good !

    seizure_samples = sample_from_model("results\++ 2 Unet1D attn mlp - pred_x0\model-15.pt", num_samples=1000, label=1, output_directory="data/synthetic/best/fnsz/")
    healthy_samples = sample_from_model("results\++ 2 Unet1D attn mlp - pred_x0\model-15.pt", num_samples=1000, label=0, output_directory="data/synthetic/best/non_seizure/")
    
    seizure_samples = sample_from_model("results\+ comb long 2 class Unet1D attn mlp - pred_x0\model-50.pt", num_samples=1000, label=1, output_directory="data/synthetic/noisy/fnsz/")
    healthy_samples = sample_from_model("results\+ comb long 2 class Unet1D attn mlp - pred_x0\model-50.pt", num_samples=1000, label=0, output_directory="data/synthetic/noisy/non_seizure/")

    seizure_samples = sample_from_model("results\++ long 2 class Unet1D attn mlp - pred_x0\model-50.pt", num_samples=1000, label=1, output_directory="data/synthetic/worst/fnsz/")
    healthy_samples = sample_from_model("results\++ long 2 class Unet1D attn mlp - pred_x0\model-50.pt", num_samples=1000, label=0, output_directory="data/synthetic/worst/non_seizure/")
    
    exit()

    seizure_samples = sample_from_model(model, num_samples=5, label=1, output_directory=None)
    healthy_samples = sample_from_model(model, num_samples=5, label=0, output_directory=None)

    plt.figure(figsize=(10, 20)) 
    for i in range(4):
        plt.subplot(4, 2, 2 * i + 1)
        plt.plot(healthy_samples[i].squeeze().cpu().numpy()[50:-50])
        plt.title('Healthy Sample ' + str(i + 1))

        plt.subplot(4, 2, 2 * i + 2)
        plt.plot(seizure_samples[i].squeeze().cpu().numpy()[50:-50])
        plt.title('Seizure Sample ' + str(i + 1))

    plt.tight_layout()
    plt.show()