# MunjeLumenDS2025 Melanoma detection
This project is designed to provide accessible melanoma detection from images, using lightweight models optimized for running on mobile or edge devices.

Training/evaluating was done with the following specs:
```YAML
CPU: Ryzen 5 5600
GPU: GTX 1070 8 GB VRAM
RAM: 48GB 3200mhz CL16
```
But it has been tested to work on the minimum:
```YAML
CPU: Macbook Intel, Ryzen 4500U
RAM: 16GB (for the devcontainer), 8GB for native setup
ROM: On HDD
GPU: Integrated graphics
```

## Clone the repo
Clone the repository using the following command in any terminal where git is available:
```bash
git clone https://github.com/pipstur/MunjeLumenDS2025.git
```

## Hosted on Streamlit!
The infrastructure is hosted on the streamlit platform on this [URL](https://melanomdetection.streamlit.app/), and single image inferencing can be done there. The current best models are deployed there.

## How it works
### Training and testing pipeline
1. Preprocessing
    - For training, the models use a dataset which is a combination of ISIC 2019 and ISIC 2020 datasets.
    - In preprocessing, the images are put through resizing, padding and CLAHE (Contrast Limited Adaptive Histogram Equalization).
2. Data splitting
    - The dataset is divided into a training set, and an independent test set.
    - The training dataset is split into 5 folds.
3. Model Training
    - Data augmentation is applied in training.
    - MobileNetV3 and EfficientFormerV2 architectures are trained on all 5 fold combinations.
    - Each combination yields 5 models, which are used in the final ensemble, during the inferencing.

![Training and Testing Pipeline Diagram](https://i.imgur.com/z8f0V8D.png)

### Inferencing pipeline
1. Preprocess an image using resizing, padding and CLAHE.
2. Inferencing through an ensemble of lightweight neural networks (MobileNetV3, EfficientFormerV2).
3. Final prediction using averaging of confidence scores.
4. Deployement on Streamlit cloud, or local inference possibility.

![Inference Pipeline Diagram](https://i.imgur.com/3rTExgB.png)

## Other README files
Consult the `readme/` folder for other information about the repository, such as:
- Setting up the environment
- Developer rules for git flow
- Getting the data
- Generating datasets for training/testing/validation
- Training the models
- Evaluating the models
- Getting `.onxx` models for inferencing
- Running inference using `onnx-runtime`
- Access to notebooks for exploration and visualization
- etc.

## Documentation
The `documentation` folder contains detailed PDFs covering key aspects of the project, including:

- Technical specifications
- Methodology overview
- Exploratory Data Analysis (EDA) report
- Model evaluation report
- Fairness analysis report


## Future work
1. Explore model optimization methods (Quantization, Model Pruning).
2. Explore larger or different architectures (ResNet, ConvNext).
3. Explore additional preprocessing techniques.
4. Explore additional augmentation techniques.
