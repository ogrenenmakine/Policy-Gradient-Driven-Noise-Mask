## Official Repo for `Policy Gradient-Driven Noise Mask` research article

This repository contains the official implementation of the "Policy Gradient-Driven Noise Mask" research article. The code is based on modifications to the PyTorch vision classification reference scripts.
`https://github.com/pytorch/vision/tree/main/references/classification`

You can find the preprint paper here: [Policy Gradient-Driven Noise Mask](https://arxiv.org/abs/2406.14568)

If you use this code in your research, please cite our paper:

```
@article{yavuz2024policy,
  title={Policy Gradient-Driven Noise Mask},
  author={Yavuz, Mehmet Can},
  journal={arXiv preprint arXiv:2406.14568},
  year={2024},
  eprint={2406.14568},
  archivePrefix={arXiv},
  primaryClass={eess.IV}
}
```

## Abstract

Deep learning classifiers face significant challenges when dealing with heterogeneous multi-modal and multi-organ biomedical datasets. This research proposes a novel pretraining pipeline that generates conditional noise masks to improve performance on these complex datasets.

### Key Features

- Dual-component system:
  1. Lightweight policy network for sampling conditional noise
  2. Classifier network
- Policy network trained using the reinforce algorithm
- Image-specific noise masks for classifier regularization during pretraining
- Policy network used only for obtaining an intermediate ("heated") model
- Direct comparison between baseline and noise-regularized models during inference

### Results

Experiments conducted on RadImageNet datasets demonstrate that fine-tuning the intermediate models consistently outperforms conventional training algorithms in:
- Classification tasks
- Generalization to unseen concept tasks


#### Installation

1. Clone the repository:
```bash
git clone https://github.com/ogrenenmakine/Policy-Gradient-Driven-Noise-Mask
cd policy-gradient-noise-mask
```

#### Download Weights
To download the pretrained weights:

```bash
cd _weights
git lfs install
git clone git@hf.co:ogrenenmakine/Policy-Gradient-Noise-Mask
```

### Training
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

#### Training with Policy-Gradient Noise Mask

```bash
cd _policy_gradient
torchrun --nproc_per_node=8 --standalone train.py \
    --data-path /path/to/radimagenet \
    --workers 16 \
    --model resnet50 \
    --batch-size 32 \
    --grid 64 \
    --kernel 13 \
    --sigma 6 \
    --sync-bn \
    --output-dir _gradp
```

#### Finetuning the Intermediate Model

```bash
cd _normal
torchrun --nproc_per_node=4 --standalone train.py \
    --data-path /path/to/radimagenet \
    --model resnet540 \
    --epochs 10 \
    --workers 16 \
    --num-classes 165 \
    --batch-size 64 \
    --lr 0.001 \
    --sync-bn \
    --lr-warmup-epochs 7 \
    --lr-warmup-method linear \
    --finetune ../_weights/resnet10t_gradientp_RIN_64_64_k13_s6.pth \
    --output-dir .
```

### Abstract

Deep learning classifiers face significant challenges when dealing with heterogeneous multi-modal and multi-organ biomedical datasets. The low-level feature distinguishability limited to imaging-modality hinders the classifiers' ability to learn high-level semantic relationships, resulting in sub-optimal performance. To address this issue, image augmentation strategies are employed as regularization techniques. While additive noise input during network training is a well-established augmentation as regularization method, modern pipelines often favor more robust techniques such as dropout and weight decay. This preference stems from the observation that combining these established techniques with noise input can adversely affect model performance.

In this study, we propose a novel *pretraining* pipeline that learns to generate conditional noise mask specifically tailored to improve performance on multi-modal and multi-organ datasets. As a reinforcement learning algorithm, our approach employs a dual-component system comprising a very light-weight policy network that learns to sample conditional noise using a differentiable beta distribution as well as a classifier network. The policy network is trained using the reinforce algorithm to generate image-specific noise masks that regularize the classifier during pretraining. A key aspect is that the policy network's role is limited to obtaining an *intermediate (or heated) model* before fine-tuning. During inference, the policy network is omitted, allowing direct comparison between the baseline and noise-regularized models.

We conducted experiments and related analyses on RadImageNet datasets. Results demonstrate that fine-tuning the intermediate models consistently outperforms conventional training algorithms on both classification and generalization to unseen concept tasks.
