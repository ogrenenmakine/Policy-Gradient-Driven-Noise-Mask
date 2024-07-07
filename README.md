### Official Repo for `Policy Gradient-Driven Noise Mask` research article

The code is modified from `https://github.com/pytorch/vision/tree/main/references/classification`

This folder contains reference training scripts. They serve as a log of how to train specific models, as provide baseline
training and evaluation scripts to quickly bootstrap research.

Except otherwise noted, all models have been trained on 8x V100 GPUs with the following parameters:

| Parameter                | value  |
| ------------------------ | ------ |
| `--batch_size`           | `32`   |
| `--epochs`               | `90`   |
| `--lr`                   | `0.1`  |
| `--momentum`             | `0.9`  |
| `--wd`, `--weight-decay` | `1e-4` |
| `--lr-step-size`         | `30`   |
| `--lr-gamma`             | `0.1`  |

### Trainin with Policy-Gradient Noise Mask
```
torchrun --nproc_per_node=8 --standalone train.py --data-path ## PATH TO RADIMAGENET --workers 16 --model resnet50 --batch-size 32 --grid 64 --kernel 13 --sigma 6 --sync-bn --output-dir _gradp
```
