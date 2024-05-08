#!/bin/bash

# python src/prepare_eeg_dataset.py --input_path data/seizure_data/train --output_path data/train_250hz_05_70_n60_CZ --duration 4 --channels "CZ" --classes "fnsz,gnsz" --frequency 250 --notch_filter 60 --lowcut 0.5 --highcut 70

python src/main.py --batch_size 64 --train_steps 7000 --data_path "train_250hz_05_70_n60_CZ" --objective "pred_v" --wandb --name "Unet1D no condition - pred_v - 250hz_05_70_n60_CZ"

python src/main.py --batch_size 64 --train_steps 7000 --data_path "train_250hz_05_40_n00_CZ" --objective "pred_v" --wandb --name "Unet1D no condition - pred_v - 250hz_05_40_n00_CZ"

# python src/main.py --batch_size 64 --train_steps 7000 --wandb --condition_type "input_concat" --name "Unet1D input concat"

# python src/main.py --batch_size 64 --train_steps 7000 --wandb --condition_type "input_add" --name "Unet1D input add"

# python src/main.py --batch_size 64 --train_steps 7000 --wandb --condition_type "attn" --name "Unet1D attn"

# python src/main.py --batch_size 64 --train_steps 7000 --wandb --condition_type "attn_mlp" --name "Unet1D attn mlp"

# python src/main.py --batch_size 64 --train_steps 7000 --wandb --condition_type "attn_embed" --name "Unet1D attn embed"

# python src/main.py --batch_size 64 --train_steps 7000 --wandb --condition_type "time_mlp" --name "Unet1D time mlp"