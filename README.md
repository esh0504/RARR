# RARR

**RARR: Real-time Attention-Driven Rain Removal with Hierarchical Scale-aware Efficient Network**

Official PyTorch implementation for training and testing.

## Requirements

```bash
pip install torch torchvision
pip install kornia tqdm scikit-image yacs opencv-python natsort wandb
```

## Dataset

Prepare datasets under `Datasets/` with the following structure:

```
Datasets/
  Rain200L/
    train/
      input/
      target/
    test/
      input/
      target/
```

Update paths in `Configs/train.yml` and `Configs/test.yml` as needed.

## Training

```bash
python train.py --config Configs/train.yml
```

Checkpoints are saved to `checkpoints/<EXP_NAME>/Deraining/models/RARR/`.

## Testing

```bash
python test.py --config Configs/test.yml
```

Set `ROOT.DIR`, `EXP_NAME`, and `ROOT.WEIGHT` in `Configs/test.yml` to point to your trained checkpoint.

## Project Structure

```
Configs/       # YAML configs for train/test
data/          # Dataset loaders
evaluate/      # PSNR/SSIM evaluation
losses/        # CharbonnierLoss, GDLoss
models/        # RARR network
utils/         # Utilities
train.py       # Training script
test.py        # Testing script
```
