# MunjeLumenDS2025 Melanoma detection
All of the code, visualizations and other things that are important to the entire solution can be found in this repository. The documentation will detail the different aspects of the entire pipeline.

## Hosted on Streamlit!
The infrastructure is hosted on the streamlit platform on this [URL](https://melanomdetection.streamlit.app/), and single image inferencing can be done there. The current best models are deployed there.

## Standards of committing and branching on the repository
1. Developers create a feature branch from main.
```bash
git checkout -b name/feature
```
Example:
```bash
git checkout -b vojislav/initial-setup
```

2. Developers commit to this branch in the following manner:
```bash
git commit -m "action: short description"
```
Example:
```bash
git commit -m "add: initial infrastructure for the project"
```
The actions that should generally be used are: `add`, `update`, `fix`. Removing something constitutes updating it, so give an additional comment in that case.

3. When work is done (everything that's improtant for the feature is committed), they create a pull request.
```bash
git push origin name/feature
```
Example:
```bash
git push origin vojislav/initial-setup
```
4. CI/CD runs status checks (linting, tests, etc).
5. Code is reviewed and approved.
- This is important because the developers need to be up to date with what is being done on the project.
6. The PR is merged using rebase and fast-forward, keeping a linear history. When merging the pull request, you should select the option to `squash and merge`.
- The source branch can be deleted when merged.
7. No merge commits, ensuring a clean Git history.
8. When you are finished, to update the remote repo with your own:
```bash
git checkout main
git fetch --prune
git pull --rebase
```

General adding, committing tips:
- Use `git status` a lot to see what you're working with.
- Use `git tree` to see what branch you're checked out to (as well as the commit history), so there is no mixup to what branch it's being committed to.

## Visual Studio Code setup
I suggest installing the following extensions, and configuring them in the settings:
- Black formatter, then go into VS Code settings > As a Default formatter add Black formatter > Search for Black > To `Black-formatter: Args` add: `--line-length=99`.
- Flake8, then go into VS Code settings > Search for Flake8 > For `Flake8: Import Strategy` put `fromEnvironment`.
- isort, then go into VS Code settings > Search for Flake8 > For `isort: Import Strategy` put `fromEnvironment`.
- Python Extension Pack is good too.
- RainbowCSV for easier viewing of `.csv` files.
- vscode-pdf for easier viewing of `.pdf` files.

## 1. Repository setup

### Virtual environment setup
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
1. For the requirements, run the following command:
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.txt
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
    --split-type kfold
```

2. Train-val-test split dataset
```bash
python training/src/train_utils/dataset_preparation/dataset_preparation.py \
    --csv-path data/get_data/merged_labels.csv \
    --images-dir data/get_data/images/ \
    --output-dir data/train_val_test/ \
    --image-size 224 224 \
    --seed 27 \
    --split-type train-val-test
```

3. Final training dataset
```bash
python training/src/train_utils/dataset_preparation/dataset_preparation.py \
    --csv-path data/get_data/merged_labels.csv \
    --images-dir data/get_data/images/ \
    --output-dir data/final_train/ \
    --image-size 224 224 \
    --seed 27 \
    --split-type train
```

### 1.2. Training models
For the entire training pipeline the following technologies are used:
1. PyTorch Lightning for reproducibility, streamlining of code writing and debloating.
2. Hydra for instantiating objects and experiment logging.
3. Tensorboard / MLFlow for result logging.

#### 1.2.1. Training scripts
1. Setup the config files

Datamodule
```YAML
_target_: training.src.datamodules.datamodule.MelanomaDataModule
data_dir: ${paths.data_dir}/dataset/ # Set the path to the data directory here
dirs: ["train", "val", "test"]
batch_size: 128
imbalanced_sampling: true
num_workers: 6 # If you have a weaker cpu keep this at 4-6
tile_size: [224, 224]
pin_memory: True
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

## 2. Running the Streamlit app locally
The streamlit app can be run locally, if you choose to iterate over it, this is done by the following command:
```bash
streamlit run app/streamlit_app.py
```

## 3. Inference script
The inference script takes models from the default `models/` folder, and the images from the assigned folder, does the inferencing on each image, aggregating the results from all of the models present in the folder and outputs a `results.csv` file as described in the competition specifications.

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

Example script run:

```bash
python inference/inference.py \
--input-folder data/test/ \
--models-folder models/ \
--output-csv results.csv \
--soft-vote
```
