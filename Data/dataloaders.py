import numpy as np
import random
import multiprocessing
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils import data
import glob
from Data.dataset import SegDataset

def split_ids(len_ids):
    """
    Splits the dataset indices into training and test sets based on a 90/10 ratio.

    Args:
        len_ids (int): Total number of samples in the dataset.

    Returns:
        tuple: Two lists containing training and test indices.
    """
    train_size = int(round((90 / 100) * len_ids))
    test_size = int(round((10 / 100) * len_ids))

    train_indices, test_indices = train_test_split(
        np.linspace(0, len_ids - 1, len_ids).astype("int"),
        test_size=test_size,
        random_state=40,
    )

    print('Test indices are: ', test_indices)
    return train_indices, test_indices

def get_dataloaders(
    input_paths_b,
    input_paths_a,
    target_paths,
    test_input_paths_b,
    test_input_paths_a,
    test_target_paths,
    batch_size=None,
    test_only=False,
    img_size=224,
    margin=4
):
    """
    Creates data loaders for training and testing datasets for segmentation tasks.

    Args:
        input_paths_b (list): Paths to images taken before ablation (training set).
        input_paths_a (list): Paths to images taken after ablation (training set).
        target_paths (list): Paths to labeled target images (training set).
        test_input_paths_b (list): Paths to images taken before ablation (test set).
        test_input_paths_a (list): Paths to images taken after ablation (test set).
        test_target_paths (list): Paths to labeled target images (test set).
        batch_size (int, optional): Batch size for training. Default is None.
        test_only (bool, optional): If True, only create a test data loader. Default is False.
        img_size (int, optional): Size to resize images to. Default is 224.
        margin (int, optional): Margin for bounding box around labeled points. Default is 4.

    Returns:
        tuple: Training and test data loaders (or just the test data loader if test_only=True).
    """

    if test_only:
        # Transformations for test inputs and targets
        transform_input4test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize((img_size, img_size), antialias=True),
                transforms.Normalize((0.04942362), (0.13466596)),
            ]
        )

        transform_target = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((img_size, img_size))]
        )

        # Create the test dataset and data loader
        test_dataset = SegDataset(
            input_paths_b=test_input_paths_b,
            input_paths_a=test_input_paths_a,
            target_paths=test_target_paths,
            transform_input=transform_input4test,
            transform_target=transform_target,
            margin=margin
        )
        test_dataloader = data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        return None, test_dataloader

    else:
        # Transformations for training inputs
        transform_input4train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize((img_size, img_size), antialias=True),
                transforms.Normalize((0.04942362), (0.13466596)),
            ]
        )

        # Transformations for test inputs
        transform_input4test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize((img_size, img_size), antialias=True),
                transforms.Normalize((0.04942362), (0.13466596)),
            ]
        )

        # Transformations for targets
        transform_target = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((img_size, img_size), antialias=True)]
        )

        # Create training and test datasets
        train_dataset = SegDataset(
            input_paths_b=input_paths_b,
            input_paths_a=input_paths_a,
            target_paths=target_paths,
            transform_input=transform_input4train,
            transform_target=transform_target,
            hflip=False,
            vflip=False,
            affine=False,
            margin=margin
        )

        test_dataset = SegDataset(
            input_paths_b=test_input_paths_b,
            input_paths_a=test_input_paths_a,
            target_paths=test_target_paths,
            transform_input=transform_input4test,
            transform_target=transform_target,
            margin=margin
        )

        # Create data loaders for training and testing
        train_dataloader = data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        test_dataloader = data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

    return train_dataloader, test_dataloader
