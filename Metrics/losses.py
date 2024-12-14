import torch
import torch.nn as nn

class SoftDiceLoss(nn.Module):
    """
    Implements the Soft Dice Loss function, commonly used in image segmentation tasks to handle imbalanced datasets.

    Args:
        smooth (float, optional): A smoothing constant added to the numerator and denominator to avoid division by zero. Default is 1.
    """
    def __init__(self, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Computes the Soft Dice Loss between predicted logits and ground truth targets.

        Args:
            logits (torch.Tensor): Predicted values, typically the raw output from the model.
            targets (torch.Tensor): Ground truth binary masks.

        Returns:
            torch.Tensor: The computed Soft Dice Loss.
        """
        num = targets.size(0)  # Batch size

        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors for each sample in the batch
        m1 = probs.view(num, -1)  # Flattened predictions
        m2 = targets.view(num, -1)  # Flattened ground truth

        # Compute the intersection between predictions and ground truth
        intersection = m1 * m2

        # Compute the Dice score for the batch
        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )

        # Average the scores across the batch and convert to loss
        score = 1 - score.sum() / num

        return score
