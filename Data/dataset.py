import random
from skimage.io import imread, imsave
import torch
from torch.utils import data
import torchvision.transforms.functional as TF
import glob
import os
import json
import numpy as np
import cv2
from torchvision import transforms

class SegDataset(data.Dataset):
    """
    A dataset class for segmentation tasks, specifically for processing images before and after ablation, along with labeled targets.

    Args:
        input_paths_b (list): Paths to images taken before ablation.
        input_paths_a (list): Paths to images taken after ablation.
        target_paths (list): Paths to labeled target images.
        transform_input (callable, optional): Transformations applied to the input images.
        transform_target (callable, optional): Transformations applied to the target images.
        hflip (bool, optional): Apply horizontal flip augmentation if True.
        vflip (bool, optional): Apply vertical flip augmentation if True.
        affine (bool, optional): Apply random affine transformations if True.
        margin (int, optional): Margin added to the bounding box around labeled points.
    """
    def __init__(
        self,
        input_paths_b: list,
        input_paths_a: list,
        target_paths: list,
        transform_input=None,
        transform_target=None,
        hflip=False,
        vflip=False,
        affine=False,
        margin=25,
    ):
        self.input_paths_b = input_paths_b
        self.input_paths_a = input_paths_a
        self.target_paths = target_paths
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.hflip = hflip
        self.vflip = vflip
        self.affine = affine
        self.margin = margin

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        print(f'len(self.input_paths_b): {len(self.input_paths_b)}')
        return len(self.input_paths_b)

    def resize2SquareKeepingAspectRation(img, size, interpolation=cv2.INTER_AREA):
        """
        Resizes an image to a square while keeping its aspect ratio by padding.

        Args:
            img (np.ndarray): The input image.
            size (int): Desired output size (width and height).
            interpolation: Interpolation method for resizing.

        Returns:
            np.ndarray: The resized square image.
        """
        h, w = img.shape[:2]
        c = None if len(img.shape) < 3 else img.shape[2]
        if h == w:
            return cv2.resize(img, (size, size), interpolation)
        dif = max(h, w)
        x_pos = (dif - w) // 2
        y_pos = (dif - h) // 2
        mask = np.zeros((dif, dif, c), dtype=img.dtype) if c else np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w] = img
        return cv2.resize(mask, (size, size), interpolation)

    def __getitem__(self, index: int):
        """
        Fetches a single data sample, applies transformations, and returns it.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            tuple: Transformed input data, target, and associated metadata.
        """
        # Load paths for before and after ablation images and the target label
        input_ID_b = self.input_paths_b[index]
        input_ID_a = os.path.join(self.input_paths_a, os.path.split(input_ID_b)[1].replace('b', 'a'))
        target_ID = os.path.join(self.target_paths, os.path.split(input_ID_b)[1].replace('b', 'a').split('.')[0] + '.png')
        json_path = os.path.join(self.target_paths, os.path.split(input_ID_b)[1].replace('b', 'a').split('.')[0] + '.json')

        print(f'input_ID_b in getitem is: {input_ID_b}')

        # Read the input and target images
        xb = imread(input_ID_b)
        xa = imread(input_ID_a)
        try:
            y = imread(target_ID)
        except:
            y = []

        # Load JSON file for labeled points
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"No JSON file found for {input_ID_a}")

        points = np.asarray(data['shapes'][0].get('points'))
        mins = np.min(points, axis=0).astype('int32') - self.margin
        maxs = np.max(points, axis=0).astype('int32') + self.margin

        # Crop images and labels using bounding box defined by points
        xb = xb[mins[1]:maxs[1], mins[0]:maxs[0], :]
        xa = xa[mins[1]:maxs[1], mins[0]:maxs[0], :]
        y = y[mins[1]:maxs[1], mins[0]:maxs[0], :]

        # Normalize target values to binary labels (0 for not ablated, 1 for ablated)
        y = (y[:, :, 0] / y[:, :, 0].max()) * 255
        y[y <= 128] = 0  # Not ablated
        y[y > 128] = 1   # Ablated

        # Add intensity to input images (optional preprocessing step)
        xb = np.clip(xb + 100, 0, 255)
        xa = np.clip(xa + 100, 0, 255)

        # Apply input and target transformations
        xb = self.transform_input(xb / 255)
        xa = self.transform_input(xa / 255)
        y = self.transform_target(y)

        # Apply data augmentations if enabled
        if self.hflip and random.uniform(0.0, 1.0) > 0.5:
            xb = TF.hflip(xb)
            xa = TF.hflip(xa)
            y = TF.hflip(y)

        if self.vflip and random.uniform(0.0, 1.0) > 0.5:
            xb = TF.vflip(xb)
            xa = TF.vflip(xa)
            y = TF.vflip(y)

        if self.affine:
            angle = random.uniform(-180.0, 180.0)
            h_trans = random.uniform(-352 / 8, 352 / 8)
            v_trans = random.uniform(-352 / 8, 352 / 8)
            scale = random.uniform(0.5, 1.5)
            shear = random.uniform(-22.5, 22.5)
            xb = TF.affine(xb, angle, (h_trans, v_trans), scale, shear, fill=-1.0)
            xa = TF.affine(xa, angle, (h_trans, v_trans), scale, shear, fill=-1.0)
            y = TF.affine(y, angle, (h_trans, v_trans), scale, shear, fill=0.0)

        # Compute the difference between "after" and "before" images
        diff = xa - xb
        diff[diff < 0] = 0

        # Concatenate input channels: before, after, and the difference
        x = torch.cat((xb, xa, diff), dim=0)

        return x.float(), y.float(), input_ID_a, target_ID, json_path
