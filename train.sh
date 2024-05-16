#!/bin/bash

# python src/prepare_eeg_dataset.py --input_path data/seizure_data/train --output_path data/train_250hz_05_40_n00_CZ --duration 4 --channels "CZ" --classes "fnsz,gnsz" --frequency 250--lowcut 0.5 --highcut 40


# python src/main.py --model custom_Unet1D --diffuser custom_GaussianDiffusion1D --batch_size 64 --train_steps 15000 --data_path "train_250hz_05_40_n00_CZ" --objective "pred_x0" --wandb --condition --condition_type "input_concat" --name "Unet1D input concat - pred_x0"

# python src/main.py --model custom_Unet1D --diffuser custom_GaussianDiffusion1D --batch_size 64 --train_steps 15000 --data_path "train_250hz_05_40_n00_CZ" --objective "pred_x0" --wandb --condition --condition_type "input_add" --name "Unet1D input add - pred_x0"

# python src/main.py --model custom_Unet1D --diffuser custom_GaussianDiffusion1D --batch_size 64 --train_steps 15000 --data_path "train_250hz_05_40_n00_CZ" --objective "pred_x0" --wandb --condition --condition_type "attn" --name "Unet1D attn - pred_x0"

# python src/main.py --model custom_Unet1D --diffuser custom_GaussianDiffusion1D --batch_size 64 --train_steps 15000 --data_path "train_250hz_05_40_n00_CZ" --objective "pred_x0" --wandb --condition --condition_type "attn_mlp" --name "Unet1D attn mlp - pred_x0"

# python src/main.py --model custom_Unet1D --diffuser custom_GaussianDiffusion1D --batch_size 64 --train_steps 15000 --data_path "train_250hz_05_40_n00_CZ" --objective "pred_x0" --wandb --condition --condition_type "attn_embed" --name "Unet1D attn embed - pred_x0"

# python src/main.py --model custom_Unet1D --diffuser custom_GaussianDiffusion1D --batch_size 64 --train_steps 15000 --data_path "train_250hz_05_40_n00_CZ" --objective "pred_x0" --wandb --condition --condition_type "time_mlp" --name "Unet1D time mlp - pred_x0"


# python src/main.py --model custom_Unet1D --diffuser custom_GaussianDiffusion1D --batch_size 64 --train_steps 15000 --data_path "train_250hz_05_40_n00_CZ" --objective "pred_x0" --condition_classes 2 --wandb --condition --condition_type "time_mlp" --name "2 Unet1D time mlp - pred_x0"

# python src/main.py --model custom_Unet1D --diffuser custom_GaussianDiffusion1D --batch_size 64 --train_steps 15000 --data_path "train_250hz_05_40_n00_CZ" --objective "pred_x0" --condition_classes 2 --wandb --condition --condition_type "attn_mlp" --name "2 Unet1D attn mlp - pred_x0"

# python src/main.py --model custom_Unet1D --diffuser custom_GaussianDiffusion1D --batch_size 64 --train_steps 15000 --data_path "train_250hz_05_40_n00_CZ" --objective "pred_x0" --condition_classes 2 --wandb --condition --condition_type "attn" --name "2 Unet1D attn - pred_x0"


python src/main.py --model custom_Unet1D --diffuser custom_GaussianDiffusion1D --batch_size 128 --train_steps 50000 --data_path "train_250hz_05_40_n00_CZ" --objective "pred_x0" --condition_classes 2 --wandb --condition --condition_type "time_mlp" --name "long 2 class Unet1D time mlp - pred_x0"

python src/main.py --model custom_Unet1D --diffuser custom_GaussianDiffusion1D --batch_size 128 --train_steps 50000 --data_path "train_250hz_05_40_n00_CZ" --objective "pred_x0" --condition_classes 2 --wandb --condition --condition_type "attn_mlp" --name "long 2 class Unet1D attn mlp - pred_x0"