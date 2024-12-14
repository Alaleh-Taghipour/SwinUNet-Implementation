
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
from Models.asam import ASAM

random.seed(12)
matplotlib.use('tkagg')
torch.manual_seed(0)
np.random.seed(0)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to desired size
    transforms.ToTensor(),
])

def save_overlay(background, true_mask, pred_mask, mean=0.13466596, std=0.04942362):
    true_mask = true_mask.astype('uint8')
    pred_mask = pred_mask.astype('uint8')
    colored_true_mask = np.zeros((np.shape(true_mask)[0], np.shape(true_mask)[1], 3), dtype="uint8")
    colored_true_mask[true_mask == 255, 0] = 200
    colored_true_mask[true_mask == 255, 1] = 100
    colored_true_mask[true_mask == 255, 2] = 0

    colored_pred_mask = np.zeros((np.shape(pred_mask)[0], np.shape(pred_mask)[1], 3), dtype="uint8")
    colored_pred_mask[pred_mask == 255, 0] = 50
    colored_pred_mask[pred_mask == 255, 1] = 250
    colored_pred_mask[pred_mask == 255, 2] = 200

    added_image = cv2.addWeighted(colored_true_mask, 0.7, colored_pred_mask, 0.3, 0)

    return added_image


def mean_absolute_error(pred, target):
    # Ensure predictions are in the 0-1 range (normalize if needed)
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)
    return torch.mean(torch.abs(pred - target))



def dice_coefficient(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred >= 0.5).float()  # Threshold at 0.5
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


@torch.no_grad()
def test(model, device, test_loader, perf_measure, do_save=True, model_name="C:/Users/a3taghip/pythonProject/Tested model"):
    t = time.time()
    model.eval()
    perf_accumulator = []
    mae_accumulator = []
    dice_accumulator = []
    cnt = 0
    if not os.path.exists(model_name):
        os.mkdir(model_name)

    for batch_idx, (data, target, _, _, _) in enumerate(test_loader):
        print(f"Batch data size: {data.shape}")  # Ensure this is [batch_size, channels, height, width]
        data, target = data.to(device), target.to(device)
        output = model(data)
        print(f"Model input size: {data.shape}")
        if do_save:
            for out_id in range(output.size()[0]):
                probs = torch.sigmoid(output[out_id, 0, :, :])
                a = probs.cpu().detach().numpy()
                a = (a >= 0.7).astype(np.uint8) * 255
                imsave(os.path.join(model_name, 'pred_' + str(cnt) + '.jpg'), a)
                y = target[out_id, 0, :, :].cpu().detach().numpy()
                y = (y * 255).astype(np.uint8)
                imsave(os.path.join(model_name, 'true_' + str(cnt) + '.jpg'), y)
                cnt += 1
                background = data[out_id, 0, :, :].cpu().detach().numpy()
                overlay = save_overlay(background, y, a)
                imsave(os.path.join(model_name, 'overlay_' + str(cnt) + '.jpg'), overlay)

        perf_accumulator.append(perf_measure(output, target).item())
        mae_accumulator.append(mean_absolute_error(output, target).item())
        dice_accumulator.append(dice_coefficient(output, target).item())

        if batch_idx + 1 < len(test_loader):
            print(
                "\rTest  [{}/{} ({:.1f}%)]\tAverage Dice Score: {:.6f}\tAverage MAE: {:.6f}\tTime: {:.6f}".format(
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(dice_accumulator),
                    np.mean(mae_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest   [{}/{} ({:.1f}%)]\tAverage Dice Score: {:.6f}\tAverage MAE: {:.6f}\tTime: {:.6f}".format(
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(dice_accumulator),
                    np.mean(mae_accumulator),
                    time.time() - t,
                )
            )
    print('Performances per case: ', perf_accumulator)
    print('Mean Absolute Errors per case: ', mae_accumulator)
    print('Dice Coefficient per case: ', dice_accumulator)
    return np.mean(perf_accumulator), np.std(perf_accumulator), np.mean(mae_accumulator), np.mean(dice_accumulator)



def batch_mean_and_sd(files, files2):
    mean = np.array([0.])
    stdTemp = np.array([0.])
    std = np.array([0.])

    numSamples = len(files)

    for i in range(numSamples):
        im = cv2.imread(str(files[i]), cv2.IMREAD_GRAYSCALE)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(float) / 255.

        # for j in range(3):
        mean += np.mean(im[:, :])

    # numSamples2 = len(files2)
    # for i in range(numSamples2):
    #     im = cv2.imread(str(files2[i]), cv2.IMREAD_GRAYSCALE)
    #     # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #     im = im.astype(float) / 255.

    #     # for j in range(3):
    #     mean += np.mean(im[:,:])

    mean = (mean / (numSamples))

    print(mean)  # 0.51775225 0.47745317 0.35173384]

    for i in range(numSamples):
        im = cv2.imread(str(files[i]), cv2.IMREAD_GRAYSCALE)

        im = im.astype(float) / 255.
        # for j in range(3):
        stdTemp += ((im[:, :] - mean) ** 2).sum() / (im.shape[0] * im.shape[1])

    std = np.sqrt(stdTemp / numSamples)

    print(std)  # [0.28075549 0.25811162 0.28913701]


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.dataset.lower() == "hifu":
        test_img_path_b = args.root + "images/before/*"
        test_img_path_a = args.root + "images/after/"
        test_mask_path = args.root + "masks/"
        test_input_paths = sorted(glob.glob(test_img_path_b))



    _, test_loader = dataloaders.get_dataloaders(
        None, None, None, test_input_paths, test_img_path_a, test_mask_path, batch_size=args.batch_size,
        img_size=args.img_size, test_only=True
    )
    # mean, std = batch_mean_and_sd(input_paths, img_path_a)
    # print("mean and std: \n", mean, std)
    Dice_loss = losses.SoftDiceLoss()
    BCE_loss = nn.BCELoss()
    # Dice_loss = DiceLoss(1)
    perf = performance_metrics.DiceScore()
    if args.model == 'transunet':  # =============== TransUnet
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = 1
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
        model.load_from(weights=np.load('model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'))
        print('\n transunet is set...\n')



    elif args.model == 'swinunet':

        from Models.vision_transformer import SwinUnet as ViT_seg

        from config import get_config

        config = get_config(args)

        model = ViT_seg(config, img_size=args.img_size, num_classes=1)

        model_path = "./Trained models/swinunet_hifu.pt"

        # Load the model weights from the specified file

        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        model_state_dict = state_dict['model_state_dict']  # Access the model's state dict

        model.load_state_dict(model_state_dict)  # Load the model's state dict

        #
        # print("Model State Dict Keys:")
        # print(model.state_dict().keys())
        # print("Loaded State Dict Keys:")
        # # Load the state dictionary from the saved model file
        # state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        #
        # # Print the keys in the loaded state dictionary
        # print(state_dict.keys())
        #
        # # Print statement
        # # Print the model's state dictionary keys
        # print(model.state_dict().keys())
        #
        # print("I am using swinunet")



    else:
        model = models.FCBFormer()

        print('\n FCBFormer is set...\n')
        try:
            model_path = "./Trained models/FCBFormer_hifu.pt"
            if os.path.exists(model_path):
                ans = input(f'Do you want to load this model: {model_path} ? (y/n) ')
                if ans == 'y':
                    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
                    model.eval()
                    print('\n model loaded successfully... \n')
                else:
                    model_path = input('Enter the model path:')
                    if os.path.exists(model_path):
                        model.load_state_dict(torch.load(model_path))
                        model.eval()
                        print('\n model loaded successfully...\n')
                    else:
                        raise ValueError('The model path does not exist...')
        except Exception as e:
            print(e)
            raise ValueError('The model path does not exist or there is an error')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.mgpu == "true":
        model = nn.DataParallel(model)
    model.to(device)
    # minimizer = ASAM(optimizer, model, rho=args.rho, eta=args.eta)

    return (
        device,
        test_loader,
        perf,
        model
    )
    # return (
    #     device,
    #     test_loader,
    #     Dice_loss,
    #     BCE_loss,
    #     perf,
    #     model,
    #     optimizer,
    # )


def predict(args):
    (
        device,
        test_loader,
        perf,
        model
    ) = build(args)

    test(model, device, test_loader, perf, do_save=True)

    print('Finished...!')


def get_args():
    parser = argparse.ArgumentParser(description="Train FCBFormer on specified dataset")
    parser.add_argument("--dataset", type=str, default='hifu', choices=["hifu", "Kvasir", "CVC"])
    parser.add_argument("--model", type=str, default='swinunet', choices=["swinunet", "fcbformer", "transunet"])
    parser.add_argument("--data-root", type=str, dest="root")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-6, dest="lr")
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument(
        "--learning-rate-scheduler", type=str, default="true", dest="lrs"
    )
    parser.add_argument(
        "--learning-rate-scheduler-minimum", type=float, default=1e-5, dest="lrs_min"
    )
    parser.add_argument(
        "--multi-gpu", type=str, default="true", dest="mgpu", choices=["true", "false"]
    )
    parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    # parser.add_argument("--rho", default=0.5, type=float, help="Rho for ASAM.")
    # parser.add_argument("--eta", default=0.1, type=float, help="Eta for ASAM.")
    parser.add_argument('--cfg', type=str,
                        default='C:/Users/a3taghip/pythonProject/configs/swin_tiny_patch4_window7_224_lite.yaml',
                        metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    return parser


def main():
    parser = get_args()
    args = parser.parse_args()
    args.root = 'Data/HIFU_data/testdata/'

    predict(args)


if __name__ == "__main__":
    main()
