import torch

class DiceScore(torch.nn.Module):
    """
    Implements the Dice Score metric, commonly used to evaluate the similarity between predicted and ground truth masks
    in image segmentation tasks.

    Args:
        smooth (float, optional): A smoothing constant added to the numerator and denominator to avoid division by zero. Default is 1.
    """
    def __init__(self, smooth=1):
        super(DiceScore, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets, sigmoid=True):
        """
        Computes the Dice Score between predicted logits and ground truth targets.

        Args:
            logits (torch.Tensor): Predicted values, typically the raw output from the model.
            targets (torch.Tensor): Ground truth binary masks.
            sigmoid (bool, optional): If True, applies the sigmoid function to logits to obtain probabilities. Default is True.

        Returns:
            torch.Tensor: The computed Dice Score.
        """
        num = targets.size(0)  # Batch size

        # Apply sigmoid to convert logits to probabilities if enabled
        probs = torch.sigmoid(logits) if sigmoid else logits

        # Flatten the tensors and binarize with a threshold of 0.5
        m1 = probs.view(num, -1) > 0.5  # Binarized predictions
        m2 = targets.view(num, -1) > 0.5  # Binarized ground truth

        # Compute the intersection between predictions and ground truth
        intersection = m1 * m2

        # Compute the Dice score for the batch
        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )

        # Average the scores across the batch
        score = score.sum() / num

        return score
