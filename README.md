# Disease_prediction

Overview of the Proposed Workflow

This figure illustrates the complete pipeline used in this study for eye disease classification using deep learning and hybrid transformer-based models. The workflow is organized into four main stages:

#### Step 1: Data Collection
The models were trained and evaluated using two publicly available fundus images  datasets: the LAG Database and an Eye Diseases Classification dataset.

1. The first one is the LAG-Database 
https://github.com/smilell/AG-CNN/tree/master?tab=readme-ov-file 

2 The second dataset is the Eye Diseases Classification dataset 
https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification

#### Step 2: Data Augmentation and Preprocessing
To improve model generalization and handle data imbalance, several augmentation and preprocessing techniques are applied. These include horizontal and vertical flipping, color jitter transformations, image normalization, and resizing images to a fixed input size compatible with deep learning models.

#### Step 3: Classification Models
Three categories of models are explored:

- Base Models: Standard CNN architectures such as ResNet50, AlexNet, EfficientNetB0, VGG, InceptionV3, DenseNet121, and Swin Transformer.

- Hybrid Models: CNN–Transformer hybrids that combine convolutional feature extractors with Swin Transformer blocks, including ResNet50-Swin, EfficientNetB0-Swin, InceptionV3-Swin, DenseNet121-Swin, AlexNet-Swin, and VGG-Swin.

- Proposed Model: FusionNet-GD, which integrates multi-model feature representations to enhance discriminative power for eye disease classification.

#### Step 4: Evaluation
Model performance is evaluated using standard classification metrics: Accuracy, Precision, Recall, and F1-score, providing a comprehensive assessment of predictive effectiveness.




Tools and libraries:
1. os: Handles file paths, directories, and operating system interactions.

2. numpy (np): Performs numerical computations and array operations.

3. torch: Core PyTorch library for tensor operations and deep learning.

4. torchvision.datasets: Loads and manages image datasets (e.g., folder-based datasets).

5. torchvision.transforms (T): Applies image preprocessing and data augmentation techniques.

- torchvision.models: Provides pretrained deep learning models for computer vision.

- torch.nn (nn): Contains neural network layers and loss functions.

- torch.optim (optim): Implements optimization algorithms for training models.

- torch.utils.data.DataLoader: Handles batching, shuffling, and efficient data loading.
