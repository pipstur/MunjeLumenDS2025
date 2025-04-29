## 0. Repository setup
On top of setting up the repository, you will also need the following CUDA requirements if you want to utilize the graphics card in inference and training:
1. CUDA Toolkit 12.x
2. cuDNN 9.x
3. PATH fully set up
4. NVIDIA graphics card drivers

There are two approaches to running this project:
1. Use the `.devcontainer` to use the repository (which will handle all of the repository, environment and requirements setup).
2. Create a virtual environment locally

The first option is done automatically, Visual Studio Code will automatically detect the `devctontainer.json` file and give you an option to run inside it. This requires you have `Docker` on your system, as well as some sort of Ubuntu distro (e.g. Windows: WSL Ubuntu). This is in the cases when you're running VSCode, as this allows for extremely easy integration. If you have a lower amount of RAM (under 16 GB), this might not work well, so resort to building the environment locally. Also important to note that if you're running a GPU setup with no NVIDIA card, it might not build (check the comment in `.devcontainer/devcontainer.json` for the fix).

The second option (creating virutal env locally) requires a few steps, it will be described in the following chapter.
### 0.1. Virtual environment setup
We will use virtual environments as it is more reliable for testing.
Creating a virtual environment requires a certain version of Python, we'll work with 3.10.

1. To create and activate a virtual environment run the following, based on your operating system:
- Windows (cmd):
```bash
python3 -m venv venv
venv\Scripts\activate
```
- Windows (PowerShell):
```bash
python3 -m venv venv
venv\Scripts\Activate.ps1
```
- Windows (Git Bash):
```bash
python -m venv venv
source venv/Scripts/activate
```
- Linux/macOS:
```bash
python -m venv venv
source venv/bin/activate
```

- Note: To deactivate a virtual environment, simply run `deactivate` in the terminal.

### 0.2. Installing dependencies
1. For the training requirements, run the following command:
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements/requirements_train.txt
```

*Optional*: If you only want to run the models in inference (using the `inference/predict.py` script), do:
```bash
pip install -r requirements/requirements_inference.txt
```

2. Pre-commit install for local linting (flake8, black, isort) (Optional, Dev only):
```bash
pip install pre-commit==2.13
pre-commit install
```

3. Install Git Large file storage, for the models to be tracked
```bash
git lfs install
```

### 0.3. Visual Studio Code setup (Optional, Dev only)
I suggest installing the following extensions, and configuring them in the settings:
- Black formatter, then go into VS Code settings > As a Default formatter add Black formatter > Search for Black > To `Black-formatter: Args` add: `--line-length=99`.
- Flake8, then go into VS Code settings > Search for Flake8 > For `Flake8: Import Strategy` put `fromEnvironment`.
- isort, then go into VS Code settings > Search for Flake8 > For `isort: Import Strategy` put `fromEnvironment`.
- Python Extension Pack is good too.
- RainbowCSV for easier viewing of `.csv` files.
- vscode-pdf for easier viewing of `.pdf` files.

This ensures there's no need to run pre-commit each time (the linting happens automatically most of the time, because of the extensions), consequently making the code versioning part a little less daunting.

### 0.4. Notebooks for Exploration and Visualization
All the development and exploration notebooks have been moved to Google Colab for easier access.

You can access them here:
[📓 Open Notebooks Folder](https://drive.google.com/drive/folders/1V9zt9TOl94Q9HRKKbFNkbee14Y2-csFm?usp=sharing)

*Tip*: Right-click on any notebook in the folder and select "Open with > Google Colab" to start experimenting immediately! This keeps the repository clean and makes it easier for collaborators to run notebooks without setting up the environment locally.


## 1. Training pipeline
### 1.0. Getting the data
The data will be downloaded, extracted and put into a format that's ready to be preparated.
```bash
usage: get_datasets.py [-h] --output-dir OUTPUT_DIR

Download and process ISIC dataset.

options:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
                        Output directory for extracted data
```

Example execution:
```bash
python training/src/train_utils/dataset_preparation/get_datasets.py \
--output-dir data/get_data/
```

### 1.1. Dataset preparation

```bash
usage: dataset_preparation.py [-h] --csv-path CSV_PATH --images-dir IMAGES_DIR --output-dir OUTPUT_DIR [--image-size IMAGE_SIZE IMAGE_SIZE] --split-type
                              {train,kfold,train-val-test} [--val-split VAL_SPLIT] [--test-split TEST_SPLIT] [--seed SEED] [--overwrite] [--kfold KFOLD] [--apply-clahe]
                              [--remove-hair] [--padding]

Dataset preparation for the training of models.

options:
  -h, --help            show this help message and exit
  --csv-path CSV_PATH   Path to the CSV file with labels.
  --images-dir IMAGES_DIR
                        Directory containing input images.
  --output-dir OUTPUT_DIR
                        Directory to save resized images.
  --image-size IMAGE_SIZE IMAGE_SIZE
                        Target image size (width height). Default: 224x224
  --split-type {train,kfold,train-val-test}
                        Type of split to perform.
  --val-split VAL_SPLIT
                        Fraction of data for validation.
  --test-split TEST_SPLIT
                        Fraction of data for testing.
  --seed SEED           Random seed for dataset split.
  --overwrite, -o       Overwrite existing directory.
  --kfold KFOLD         Number of folds for K-Fold cross-validation.
  --apply-clahe         Whether to apply CLAHE enhancment.
  --remove-hair         Whether to apply hair removal.
  --padding             Whether to add padding to scaled image.
```

Running the script for dataset preparations is done as so:

1. K-fold dataset
```bash
python training/src/train_utils/dataset_preparation/dataset_preparation.py \
    --csv-path data/get_data/merged_labels.csv \
    --images-dir data/get_data/images/ \
    --output-dir data/kfold_train/ \
    --image-size 224 224 \
    --seed 27 \
    --kfold 5 \
    --split-type kfold \
    --apply-clahe \
    --padding
```

2. Train-val-test split dataset
```bash
python training/src/train_utils/dataset_preparation/dataset_preparation.py \
    --csv-path data/get_data/merged_labels.csv \
    --images-dir data/get_data/images/ \
    --output-dir data/train_val_test/ \
    --image-size 224 224 \
    --seed 27 \
    --split-type train-val-test \
    --apply-clahe \
    --padding
```

3. Final training dataset
```bash
python training/src/train_utils/dataset_preparation/dataset_preparation.py \
    --csv-path data/get_data/merged_labels.csv \
    --images-dir data/get_data/images/ \
    --output-dir data/final_train/ \
    --image-size 224 224 \
    --seed 27 \
    --split-type train \
    --apply-clahe \
    --padding
```

### 1.2. Training models
For the entire training pipeline the following technologies are used:
1. PyTorch Lightning for reproducibility, streamlining of code writing and debloating.
2. Hydra for instantiating objects and experiment logging.
3. Tensorboard for result and experiment logging.

#### 1.2.1. Training scripts
1. Setup the config files

Datamodule
```YAML
_target_: training.src.datamodules.datamodule.MelanomaDataModule
data_dir: ${paths.data_dir}/kfold_train/ # Set the path to the data directory here
dirs: ["train", "val", "test"]
batch_size: 128
imbalanced_sampling: False
num_workers: 6 # If you have a weaker cpu keep this at 4-6
tile_size: [224, 224]
pin_memory: False
grayscale: False
train_da: True
val_da: False
```

- Note: The dataset directory that you are setting in the datamodule config must match the training you're going to do on it (regular training/val or K-Fold structure)

Model
```YAML
_target_: training.src.models.mobilenetv3.MobileNetV3

optimizer:
  _target_: torch.optim.Adam
  _partial_: true
  lr: 0.001 # Set the learning rate as you see fit
  weight_decay: 0.001 # And the R2 regularization

scheduler:
  _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
  _partial_: true
  mode: min
  factor: 0.1
  patience: 5

freeze_layers: true

loss_function: focal
```

2. Run the script for training or K-Fold training
```bash
python training/src/train/train.py
python training/src/train/train_kfold.py
```

- Note: You can override parameters you set in the config files, for example change the dataset directory inside the run of the script. This works for every parameter in the configs, but to give an example: `python training/src/train/train.py datamodule.data_dir=data/other_dataset/`

#### 1.2.2. Hyperparameter optimization
The hyperparameter optimization is done using grid search (or other search supported by optuna)

To do hyperparameter optimization:
1. Set the config file up `hparams_search/grid_search.yaml`
```YAML
# @package _global_
defaults:
  - override /hydra/sweeper: basic
optimized_metric: "val/roc_auc_best"
hydra:
  mode: "MULTIRUN"

  sweeper:
    params:
      model.optimizer.lr: choice(0.1, 0.01, 0.001, 0.0001)
      datamodule.batch_size: choice(64, 128)
      model.optimizer.weight_decay: choice(0.0, 0.01, 0.001)

```
2. Run the training script
```bash
python training/src/train/train.py -m hparams_search=grid_search
python training/src/train/train_kfold.py -m hparams_search=grid_search
```

### 1.3. Evaluating models
Model evaluation can be done using the `eval.py` method, to which the checkpoint path is passed in order to load it.
```bash
python training/src/eval/eval.py ckpt_path=training/logs/train/runs/run/checkpoints/epoch_x.ckpt
```
- Note: The model to which the checkpoint is pointing to, and the model passed to the `model=` parameter must be the same.

## 2. Inference script
The inference script takes models from the default `models/` folder, and the images from the assigned folder, does the inferencing on each image, aggregating the results from all of the models present in the folder and outputs a `results.csv` file as described in the competition specifications.

### 2.1. Download the models
You can download the released models using the `download_models.py` script:

```bash
python models/download_models.py \
--repo-owner pipstur \
--repo-name MunjeLumenDS2025 --release-tag v2.0.0 \
--download-dir downloads/ \
--extract-dir models/
```

### 2.2. Running the inference script
```bash
usage: inference.py [-h] --input-folder INPUT_FOLDER [--models-folder MODELS_FOLDER] [--output-csv OUTPUT_CSV] [--save-tiles-folder SAVE_TILES_FOLDER]
                    [--soft-vote]

ONNX Model Inference with Majority Voting

options:
  -h, --help            show this help message and exit
  --input-folder INPUT_FOLDER
                        Folder containing input images
  --models-folder MODELS_FOLDER
                        Folder containing ONNX models
  --output-csv OUTPUT_CSV
                        Path to save the output CSV
  --save-tiles-folder SAVE_TILES_FOLDER
                        Folder to save preprocessed image tiles
  --soft-vote           Use soft voting for predictions
```

Example script run, assuming the images are inside the `data/test/` folder:

```bash
python inference/predict.py data/test/ results/output_csv.csv
```

*Note*: If you do not have CUDA requirements highlighted in this README, you the `onnx-runtime` library will resort back to CPU inference. It will still run, but there will be warnings for each of the models, and the inference will be much slower.

## 3. Running the Streamlit app locally
The streamlit app can be run locally, if you choose to iterate over it, this is done by the following command:
```bash
streamlit run app/streamlit_app.py
```

## 4. Segmentation and skin color estimation
### 4.1. Preparing the environment
To run segmentation of lesions, first use the following commands to create a new virtual environment (this is because Tensorflow and PyTorch have some weird CUDA conflicts):
- Windows (cmd):
```bash
python3 -m venv segmentation-venv
segmentation-venv\Scripts\activate
```
- Windows (PowerShell):
```bash
python3 -m venv segmentation-venv
segmentation-venv\Scripts\Activate.ps1
```
- Windows (Git Bash):
```bash
python -m venv segmentation-venv
source segmentation-venv/Scripts/activate
```
- Linux/macOS:
```bash
python -m venv segmentation-venv
source segmentation-venv/bin/activate
```
And then install the requirements:
```bash
pip install -r requirements/requirements_segmentation.txt
```

### 4.2. Running the segmentation script
Example run of the segmentation script which will result in a `.csv` file from which you can get the skin color information.
```bash
python training/src/train_utils/segmentation/segmentation_skin_color_analysis.py \
--csv-path data/get_data/merged_labels.csv \
--images-dir data/get_data/images/ \
--masks-dir data/segmentation_masks/ \
--csv-out-path results/skin_color_estimates.csv
```
