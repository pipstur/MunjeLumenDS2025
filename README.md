# MunjeLumenDS2025 Melanoma detection
All of the code, visualizations and other things that are important to the entire solution can be found in this repository. The documentation will detail the different aspects of the entire pipeline.

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
- The source branch will be deleted when merged.
7. No merge commits, ensuring a clean Git history.
8. When you are finished, to update the remote repo with your own:
```bash
git checkout main
git fetch --prune
git pull --rebase
```

General adding, committing tips:
- Use `git status` a lot to see what you're working with.
- Use `git tree` to see what branch you're checked out to, so there is no mixup to what branch is being committed to.

## 0. Repository setup
This part of the documentation will detail the different steps in setting up the environment in order to run this project.
### 0.1. Virtual environment setup
We will use virtual environments as it is more reliable for testing.
Creating a virtual environment requires a certain version of Python, we'll work with 3.10.

1. To create a virtual environment run the following:
`python3.10 -m venv venv`
2. Then, based on operating system, in the chosen terminal run the following:
- Windows (cmd):
`venv\Scripts\activate`
- Windows (PowerShell):
`venv\Scripts\Activate.ps1`
- Windows (Git Bash):
`source venv/Scripts/activate`
- Linux/macOS:
`source venv/bin/activate`

Note: To deactivate a virtual environment, simply run `deactivate` in the terminal.

### 0.2. Installing dependencies
1. For the requirements, run the following command:
```bash
pip install -r requirements.txt
```

2. Pre-commit install for local linting (flake8, black, isort) (Optional, Dev only):
```bash
pip install pre-commit==2.13
pre-commit install
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

### 1.1. Dataset preparation

```bash
usage: dataset_preparation.py [-h] --csv-path CSV_PATH --images-dir IMAGES_DIR --output-dir OUTPUT_DIR [--image-size IMAGE_SIZE IMAGE_SIZE] [--val-split VAL_SPLIT] [--seed SEED]
                              [--overwrite]

Resize images and split dataset.

options:
  -h, --help            show this help message and exit
  --csv-path CSV_PATH   Path to the CSV file with labels.
  --images-dir IMAGES_DIR
                        Directory containing input images.
  --output-dir OUTPUT_DIR
                        Directory to save resized images.
  --image-size IMAGE_SIZE IMAGE_SIZE
                        Target image size (width height). Default: 224x224
  --val-split VAL_SPLIT
                        Fraction of data for validation (default: 0.2)
  --seed SEED           Random seed for dataset split (default: 27)
  --overwrite, -o       Overwrite existing directory.
```

Running the script for dataset preparations is done as so, if you placed the original dataset into the `data/` folder:

```bash
python training/src/train_utils/dataset_preparation/dataset_preparation.py \
    --csv-path data/train/ISIC_2020_Training_GroundTruth.csv \
    --images-dir data/train/images/ \
    --output-dir data/train/resized/ \
    --image-size 224 224 \
    --val-split 0.2 \
    --seed 27 \
    --kfold 5
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
batch_size: 128
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
optimized_metric: "val/acc_best"
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
