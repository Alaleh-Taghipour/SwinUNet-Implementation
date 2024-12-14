import os
import glob
import argparse
import numpy as np

from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
from skimage.io import imread
from skimage.transform import resize


def eval_predictions(args):
    """
    Evaluate the model predictions by computing metrics such as Dice, IoU, Precision, and Recall.

    Args:
        args (Namespace): Parsed command-line arguments.
    """
    # Retrieve prediction files
    prediction_files = sorted(glob.glob(os.path.join(args.predictions_path, "*")))

    # Retrieve ground truth mask files
    target_paths = sorted(glob.glob(os.path.join(args.data_root, "masks", "*")))

    dice_scores, iou_scores, precision_scores, recall_scores = [], [], [], []

    for i, (pred_path, target_path) in enumerate(zip(prediction_files, target_paths)):
        pred = np.ndarray.flatten(imread(pred_path) / 255 > 0.5)
        gt = resize(imread(target_path), (352, 352), anti_aliasing=False) > 0.5

        if len(gt.shape) == 3:
            gt = np.mean(gt, axis=2)
        gt = np.ndarray.flatten(gt)

        # Compute metrics
        dice_scores.append(f1_score(gt, pred))
        iou_scores.append(jaccard_score(gt, pred))
        precision_scores.append(precision_score(gt, pred))
        recall_scores.append(recall_score(gt, pred))

        print(
            f"Test: [{i + 1}/{len(prediction_files)}]\t"
            f"Dice: {np.mean(dice_scores):.6f}, IoU: {np.mean(iou_scores):.6f}, "
            f"Precision: {np.mean(precision_scores):.6f}, Recall: {np.mean(recall_scores):.6f}",
            end="\r" if i + 1 < len(prediction_files) else "\n"
        )


def get_args():
    """
    Parse command-line arguments for evaluation.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate model predictions.")
    parser.add_argument(
        "--predictions-path",
        type=str,
        required=True,
        help="Path to the directory containing model prediction files."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing ground truth mask files."
    )
    return parser.parse_args()


def main():
    args = get_args()
    eval_predictions(args)


if __name__ == "__main__":
    main()
