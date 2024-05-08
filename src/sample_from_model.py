import torch
import matplotlib.pyplot as plt

from model import Unet1D, GaussianDiffusion1D


def standarize(t):
    return (t - t.mean()) / t.std()


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    backbone = Unet1D(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        condition=True,
        condition_type='time_mlp',  # 'input_concat', 'input_add', 'attn', 'attn_mlp', 'attn_embed', 'time_mlp'
    )

    model = GaussianDiffusion1D(
        backbone,
        seq_length=1000,
        timesteps=1000,
        objective='pred_v'
    )

    model.load_state_dict(torch.load("tests/results/model-7.pt")['model'])

    model.eval().cuda()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    with torch.no_grad():
        samples = model.sample(batch_size=4)

    plt.figure(figsize=(10, 5))
    for n,i in enumerate(samples.cpu()): 
        plt.plot(i.squeeze())
        # plt.show()
    plt.savefig(f"test.png")