import sys
import os
import argparse
import time
import numpy as np
import glob

import torch
import torch.nn as nn

from Data import dataloaders
from Models import models
from Metrics import performance_metrics
from Metrics import losses
import shutil, cv2
import matplotlib
from skimage.io import imsave
import random
from transunet_networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from transunet_networks.vit_seg_modeling import VisionTransformer as ViT_seg
from utils import DiceLoss

random.seed(12)
matplotlib.use('tkagg')
torch.manual_seed(0)
np.random.seed(0)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def save_overlay(background, true_mask, pred_mask):
    """
    Generate an overlay image by combining true and predicted masks.

    Args:
        background (np.ndarray): Background image.
        true_mask (np.ndarray): Ground truth mask.
        pred_mask (np.ndarray): Predicted mask.

    Returns:
        np.ndarray: Combined overlay image.
    """
    true_mask = true_mask.astype('uint8')
    pred_mask = pred_mask.astype('uint8')

    colored_true_mask = np.zeros((true_mask.shape[0], true_mask.shape[1], 3), dtype="uint8")
    colored_true_mask[true_mask == 255] = [200, 100, 0]

    colored_pred_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype="uint8")
    colored_pred_mask[pred_mask == 255] = [50, 250, 200]

    return cv2.addWeighted(colored_true_mask, 0.7, colored_pred_mask, 0.3, 0)

def train_epoch(model, device, train_loader, optimizer, epoch, Dice_loss, BCE_loss):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): Model to be trained.
        device (torch.device): Device to use (CPU or GPU).
        train_loader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer for the model.
        epoch (int): Current epoch number.
        Dice_loss (Loss): Dice loss function.
        BCE_loss (Loss): Binary Cross-Entropy loss function.

    Returns:
        float: Average loss over the epoch.
    """
    t = time.time()
    model.train()
    loss_accumulator = []

    for batch_idx, (data, target, _, _, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = Dice_loss(output, target) + BCE_loss(torch.sigmoid(output), target)
        loss.backward()
        optimizer.step()
        loss_accumulator.append(loss.item())

        progress = "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
            epoch,
            (batch_idx + 1) * len(data),
            len(train_loader.dataset),
            100.0 * (batch_idx + 1) / len(train_loader),
            loss.item(),
            time.time() - t,
        )
        print(progress, end="" if batch_idx + 1 < len(train_loader) else "\n")

    return np.mean(loss_accumulator)

@torch.no_grad()
def test(model, device, test_loader, epoch, perf_measure, do_save=True):
    """
    Evaluate the model on the test dataset.

    Args:
        model (torch.nn.Module): Model to be evaluated.
        device (torch.device): Device to use (CPU or GPU).
        test_loader (DataLoader): Test data loader.
        epoch (int): Current epoch number.
        perf_measure (Callable): Performance metric function.
        do_save (bool, optional): Flag to save predictions. Defaults to True.

    Returns:
        tuple: Mean and standard deviation of the performance metric.
    """
    t = time.time()
    model.eval()
    perf_accumulator = []
    cnt = 0

    for batch_idx, (data, target, _, _, _) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)

        if do_save:
            for out_id in range(output.size()[0]):
                probs = torch.sigmoid(output[out_id, 0, :, :])
                pred_mask = (probs.cpu().detach().numpy() >= 0.7).astype(np.uint8) * 255
                true_mask = (target[out_id, 0, :, :].cpu().detach().numpy() * 255).astype(np.uint8)

                imsave(f'./results/pred_{cnt}.jpg', pred_mask)
                imsave(f'./results2/true_{cnt}.png', true_mask)

                overlay = save_overlay(data[out_id, 0, :, :].cpu().detach().numpy(), true_mask, pred_mask)
                imsave(f'./results2/overlay_{cnt}.jpg', overlay)
                cnt += 1

        perf_accumulator.append(perf_measure(output, target).item())
        progress = "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
            epoch,
            batch_idx + 1,
            len(test_loader),
            100.0 * (batch_idx + 1) / len(test_loader),
            np.mean(perf_accumulator),
            time.time() - t,
        )
        print(progress, end="" if batch_idx + 1 < len(test_loader) else "\n")

    return np.mean(perf_accumulator), np.std(perf_accumulator)

def batch_mean_and_sd(files):
    """
    Compute the mean and standard deviation of a batch of images.

    Args:
        files (list): List of image file paths.

    Returns:
        tuple: Mean and standard deviation of the images.
    """
    mean = np.array([0.])
    std_temp = np.array([0.])
    num_samples = len(files)

    for file in files:
        im = cv2.imread(file, cv2.IMREAD_GRAYSCALE).astype(float) / 255.
        mean += np.mean(im)

    mean /= num_samples

    for file in files:
        im = cv2.imread(file, cv2.IMREAD_GRAYSCALE).astype(float) / 255.
        std_temp += ((im - mean) ** 2).sum() / (im.size)

    std = np.sqrt(std_temp / num_samples)
    return mean, std

def build(args):
    """
    Build and initialize the model, data loaders, optimizer, and loss functions.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        tuple: Initialized components for training and testing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data preparation
    if args.dataset.lower() == "hifu":
        img_path_b = os.path.join(args.root, "images/before/*")
        img_path_a = os.path.join(args.root, "images/after/")
        mask_path = os.path.join(args.root, "masks/")
        input_paths = sorted(glob.glob(img_path_b))

        test_img_path_b = os.path.join(args.root, "test_data_asam/images/before/*")
        test_img_path_a = os.path.join(args.root, "test_data_asam/images/after/")
        test_mask_path = os.path.join(args.root, "test_data_asam/masks/")
        test_input_paths = sorted(glob.glob(test_img_path_b))

    train_dataloader, test_loader = dataloaders.get_dataloaders(
        input_paths, img_path_a, mask_path, test_input_paths, test_img_path_a, test_mask_path,
        batch_size=args.batch_size, img_size=args.img_size,
    )

    Dice_loss = losses.SoftDiceLoss()
    BCE_loss = nn.BCELoss()
    perf = performance_metrics.DiceScore()

    # Model selection
    if args.model == 'transunet':
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = 1
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (
                args.img_size // args.vit_patches_size, args.img_size // args.vit_patches_size)
        model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
        model.load_from(weights=np.load('model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'))

    elif args.model == 'swinunet':
        from Models.vision_transformer import SwinUnet as ViT_seg
        from config import get_config
        config = get_config(args)
        model = ViT_seg(config, img_size=args.img_size, num_classes=1).cuda()
        model.load_from(config)

    else:
        model = models.FCBFormer()
        if os.path.exists("./Trained models/FCBFormer_Hifu.pt"):
            model.load_state_dict(torch.load("./Trained models/FCBFormer_Hifu.pt"))
            model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.mgpu == "true":
        model = nn.DataParallel(model)
    model.to(device)

    return device, train_dataloader, test_loader, Dice_loss, BCE_loss, perf, model, optimizer

def train(args):
    """
    Train the model for the specified number of epochs.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    (device, train_dataloader, test_dataloader, Dice_loss, BCE_loss, perf, model, optimizer) = build(args)

    if not os.path.exists("./Trained models"):
        os.makedirs("./Trained models")

    prev_best_test = None
    if args.lrs == "true":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, min_lr=args.lrs_min if args.lrs_min > 0 else None, verbose=True
        )

    loss_values = []
    for epoch in range(1, args.epochs + 1):
        try:
            loss = train_epoch(model, device, train_dataloader, optimizer, epoch, Dice_loss, BCE_loss)
            loss_values.append(loss)

            test_measure_mean, test_measure_std = test(model, device, test_dataloader, epoch, perf)

            if args.lrs == "true":
                scheduler.step(test_measure_mean)

            if prev_best_test is None or test_measure_mean > prev_best_test:
                if test_measure_mean > 0.82:
                    print("Saving...")
                    test(model, device, test_dataloader, epoch, perf, do_save=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict() if args.mgpu == "false" else model.module.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss,
                            "test_measure_mean": test_measure_mean,
                            "test_measure_std": test_measure_std,
                        },
                        f"Trained models/{args.model}_" + args.dataset + ".pt",
                    )
                prev_best_test = test_measure_mean

        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)

    import matplotlib.pyplot as plt
    plt.plot(loss_values, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()
    plt.show()

def get_args():
    """
    Parse command-line arguments for training configuration.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train FCBFormer on specified dataset")
    parser.add_argument("--dataset", type=str, default='hifu', choices=["hifu", "Kvasir", "CVC"])
    parser.add_argument("--model", type=str, default='swinunet', choices=["fcbformer", "transunet", "swinunet"])
    parser.add_argument("--data-root", type=str, dest="root")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4, dest="lr")
    parser.add_argument('--img_size', type=int, default=224, help='Input image size.')
    parser.add_argument("--learning-rate-scheduler", type=str, default="true", dest="lrs")
    parser.add_argument("--learning-rate-scheduler-minimum", type=float, default=1e-5, dest="lrs_min")
    parser.add_argument("--multi-gpu", type=str, default="true", dest="mgpu", choices=["true", "false"])
    parser.add_argument('--n_skip', type=int, default=3, help='Number of skip connections.')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='Vision Transformer model name.')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='Patch size for Vision Transformer.')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='Path to config file.')
    parser.add_argument("--opts", nargs='+', help="Modify config options by adding 'KEY VALUE' pairs.", default=None)
    parser.add_argument('--zip', action='store_true', help='Use zipped dataset instead of folder dataset.')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'], help='Dataset caching mode.')
    parser.add_argument('--resume', help='Resume from checkpoint.')
    parser.add_argument('--accumulation-steps', type=int, help="Gradient accumulation steps.")
    parser.add_argument('--use-checkpoint', action='store_true', help="Use gradient checkpointing to save memory.")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='AMP optimization level.')
    parser.add_argument('--tag', help='Tag of experiment.')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only.')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only.')
    return parser

def main():
    """
    Main function to parse arguments and start training.
    """
    parser = get_args()
    args = parser.parse_args()
    args.epochs = 1000
    args.root = 'Data/HIFU_data/'
    train(args)

if __name__ == "__main__":
    main()
