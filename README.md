# Transfer Learning for Drone Detection using Faster R-CNN

Object detection model for identifying drones in images, built by fine-tuning a pretrained Faster R-CNN via transfer learning.

## Overview

This project applies transfer learning to a Faster R-CNN backbone to detect drones in images. Instead of training from scratch, a pretrained model is fine-tuned on a drone-specific dataset, reducing training time and data requirements while achieving strong detection performance.

## Training & Evaluation

Loss, validation mAP, and learning rate schedule over training:

![Training graphs](graphs.png)

Confusion matrix on the validation set:

![Confusion matrix](confusion_matrix.png)

Precision, recall, and F2 score:

![Scores](scores.png)

## Detection Results

![Detection result 1](test1.png)
![Detection result 2](test2.png)
![Detection result 3](test3.png)

## Contents

- `notebook.ipynb` — full pipeline: data loading, model setup, fine-tuning, evaluation, and inference visualization.

## Approach

1. **Base model**: Faster R-CNN pretrained on COCO.
2. **Transfer learning**: backbone frozen/partially fine-tuned; detection head retrained on drone images.
3. **Training**: fine-tuned on labeled drone bounding-box data.
4. **Evaluation**: inference run on test images, results visualized with predicted bounding boxes (see graphs and results above).

## Requirements

- Python 3.x
- PyTorch
- torchvision
- Jupyter Notebook

Install with:
```bash
pip install torch torchvision jupyter
```

## Usage

```bash
git clone https://github.com/OfirTzrik/Transfer-learning-for-drone-detection-using-Faster-R-CNN.git
cd Transfer-learning-for-drone-detection-using-Faster-R-CNN
jupyter notebook notebook.ipynb
```

Run the notebook cells in order to reproduce training and inference.

## License

No license specified.
