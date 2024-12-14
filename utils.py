import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks.

    Attributes:
        n_classes (int): Number of segmentation classes.
    """

    def __init__(self, n_classes):
        """
        Initialize the DiceLoss module.

        Args:
            n_classes (int): Number of segmentation classes.
        """
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        """
        Encode input tensor into a one-hot representation.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: One-hot encoded tensor.
        """
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        return torch.cat(tensor_list, dim=1).float()

    def _dice_loss(self, score, target):
        """
        Compute Dice loss for a single class.

        Args:
            score (torch.Tensor): Predicted probabilities for a class.
            target (torch.Tensor): Ground truth binary mask for a class.

        Returns:
            float: Dice loss value.
        """
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        """
        Compute Dice loss for all classes.

        Args:
            inputs (torch.Tensor): Predicted probabilities.
            target (torch.Tensor): Ground truth binary masks.
            weight (list, optional): Weights for each class. Defaults to None.
            softmax (bool, optional): Apply softmax to inputs. Defaults to False.

        Returns:
            float: Average Dice loss across all classes.
        """
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        if weight is None:
            weight = [1] * self.n_classes

        assert inputs.size() == target.size(), \
            f'Predicted {inputs.size()} and target {target.size()} shapes do not match.'

        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]

        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    """
    Calculate Dice coefficient and 95th Hausdorff distance for a single case.

    Args:
        pred (np.ndarray): Predicted binary mask.
        gt (np.ndarray): Ground truth binary mask.

    Returns:
        tuple: Dice coefficient and Hausdorff distance.
    """
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256],
                       test_save_path=None, case=None, z_spacing=1):
    """
    Test a single volume for segmentation and compute metrics.

    Args:
        image (torch.Tensor): Input image volume.
        label (torch.Tensor): Ground truth labels for the volume.
        net (torch.nn.Module): Segmentation model.
        classes (int): Number of segmentation classes.
        patch_size (list, optional): Patch size for resizing. Defaults to [256, 256].
        test_save_path (str, optional): Path to save predictions. Defaults to None.
        case (str, optional): Identifier for the volume case. Defaults to None.
        z_spacing (float, optional): Spacing in the z-dimension. Defaults to 1.

    Returns:
        list: Metrics (Dice coefficient and HD95) for each class.
    """
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice_ = image[ind, :, :]
            x, y = slice_.shape[0], slice_.shape[1]

            if x != patch_size[0] or y != patch_size[1]:
                slice_ = zoom(slice_, (patch_size[0] / x, patch_size[1] / y), order=3)

            input_tensor = torch.from_numpy(slice_).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()

            with torch.no_grad():
                outputs = net(input_tensor)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().detach().numpy()

                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out

                prediction[ind] = pred
    else:
        input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()

        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input_tensor), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))

        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))

        sitk.WriteImage(prd_itk, f'{test_save_path}/{case}_pred.nii.gz')
        sitk.WriteImage(img_itk, f'{test_save_path}/{case}_img.nii.gz')
        sitk.WriteImage(lab_itk, f'{test_save_path}/{case}_gt.nii.gz')

    return metric_list
