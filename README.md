# SwinUNet-Implementation
Implementation of SwinUNet for HIFU image segmentation tasks
This repository is an adaptation of the [HIFU_Segmentation](https://github.com/hosseinbv/HIFU_Segmentation) repository by hosseinbv, modified to work with specific datasets and tasks.

Implementation of SwinUNet for HIFU image segmentation tasks


This repository contains an implementation of **SwinUNet**, a hierarchical Transformer model for image segmentation tasks. The code has been adapted and updated to fit specific datasets and requirements for [Swin-Implementation for HIFU Images Segmentation].

# SwinUNet-Based Segmentation Framework for Ablation Detection

## Overview
This project provides an implementation of a segmentation framework leveraging SwinUNet, a powerful architecture for medical image segmentation. It is tailored for detecting ablation regions in High-Intensity Focused Ultrasound (HIFU) experiments, enabling the analysis of images captured before and after ablation. This framework handles data preprocessing, segmentation model training, and testing, ensuring a robust pipeline for research purposes.

## Features
- **Input Pair Processing**: Handles image pairs before and after ablation.
- **Mask Labeling**: Incorporates labeled masks for ablation regions.
- **Data Augmentation**: Includes flipping and affine transformations for robust training.
- **Easy Integration**: Designed for flexible integration with existing pipelines.

## Repository Structure
### Key Files and Scripts
1. **`SegDataset` (dataset_code.py)**
   - Implements a custom PyTorch dataset class for loading and processing image pairs and their corresponding masks.
   - Supports transformations, cropping, and augmentation to prepare the data for training.

2. **Preprocessing and Augmentation**
   - Includes functions for resizing images, flipping, and applying affine transformations.

3. **Training and Testing Scripts**
   - Designed to load the dataset, preprocess the data, and train a SwinUNet model for ablation detection.

## Setup and Usage
### Prerequisites
- Python 3.8+
- PyTorch 1.11.0+
- Additional dependencies listed in `requirements.txt`.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Alaleh-Taghipour/SwinUNet-Implementation.git
   cd SwinUNet-Implementation
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Organization
Organize your dataset in the following structure:
```
/Data/HIFU_data/
    test_data_asam/
        images/
            before/
                image1_b.jpg
                image2_b.jpg
                ...
            after/
                image1_a.jpg
                image2_a.jpg
                ...
        masks/
            image1_a.png
            image2_a.png
            ...
```

### Running the Code
1. **Dataset Preparation**
   - Update dataset paths in the code to point to your local dataset.

2. **Training**
   - Configure transformations and model settings in the training script.
   - Run the script to start training:
     ```bash
     python train.py
     ```

3. **Testing**
   - Load the trained model and dataset for evaluation:
     ```bash
     python test.py
     ```

### Example Output
- Input pairs (`before` and `after`) are processed, and the ablation region mask is predicted.
- Outputs include processed tensors and predictions:
  ```plaintext
  batch_idx: 0
  x.shape: torch.Size([3, 224, 224])    y.shape: torch.Size([1, 224, 224])
  input_ID_a: ['path_to_after_image']   target_ID: ['path_to_target_mask']
  json_path: ['path_to_label_json']
  ```

## Code Explanation
### Dataset Class (`SegDataset`)
The `SegDataset` class is the backbone of this framework, handling data loading, preprocessing, and augmentation.
#### Key Components:
- **Data Loading**: Reads image pairs (`before` and `after`) and corresponding labels.
- **Preprocessing**: Applies transformations such as resizing and normalization.
- **Augmentation**: Includes options for flipping and affine transformations.
- **Binary Mask Conversion**: Converts labeled masks into binary format for segmentation tasks.

#### Key Methods:
1. `__len__`: Returns the size of the dataset.
2. `__getitem__`: Loads a single sample, processes it, and applies transformations.
3. `resize2SquareKeepingAspectRation`: Resizes images while preserving aspect ratio.

## Acknowledgments
This implementation is based on the repository:
- [HIFU_Segmentation by hosseinbv](https://github.com/hosseinbv/HIFU_Segmentation)

We greatly acknowledge the original work for inspiring this framework.

## License
This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
<<<<<<< HEAD

=======
>>>>>>> 6505de6 (Add all project files and code)
