# MunjeLumenDS2025 Melanoma detection
This project is designed to provide accessible melanoma detection from images, using lightweight models optimized for running on mobile or edge devices.

Training/evaluating was done with the following specs:
```YAML
CPU: Ryzen 5 5600
GPU: GTX 1070 8 GB VRAM
RAM: 48GB 3200mhz CL16
```

## Hosted on Streamlit!
The infrastructure is hosted on the streamlit platform on this [URL](https://melanomdetection.streamlit.app/), and single image inferencing can be done there. The current best models are deployed there.

## Notebooks for Exploration and Visualization
All the development and exploration notebooks have been moved to Google Colab for easier access.

You can access them here:
[📓 Open Notebooks Folder](https://drive.google.com/drive/folders/1V9zt9TOl94Q9HRKKbFNkbee14Y2-csFm?usp=sharing)

*Tip*: Right-click on any notebook in the folder and select "Open with > Google Colab" to start experimenting immediately! This keeps the repository clean and makes it easier for collaborators to run notebooks without setting up the environment locally.

## How it works
1. Preprocess an image using resizing, padding and CLAHE.
2. Inferencing through an ensemble of lightweight neural networks (SqueezeNet1.1, ShuffleNetV2, MobileNetV3).
3. Final prediction using averaging of confidence scores.
4. Deployement on Streamlit cloud, or local inference possibility.

![Pipeline Diagram](https://i.imgur.com/fwXLSO4.png)

## Other README files
Consult the `readme/` folder for other information about the repository, such as:
- Setting up the environment
- Developer rules for git flow
- Training the models
- Evaluating the models
- Running inference using `onnx-runtime`
- etc.

## Future work
1. Explore model optimization methods (Quantization, Model Pruning).
2. Explore larger or different architectures (ResNet, ConvNext).
3. Explore additional preprocessing techniques.
4. Explore additional augmentation techniques.
