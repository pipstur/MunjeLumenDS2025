## Environment setup
### Approaches to environment setup
There are two approaches to running this project:
1. Use the `.devcontainer` to use the repository (which will handle all of the repository, environment and requirements setup).
2. Create a virtual environment locally

The first option is done automatically, Visual Studio Code will automatically detect the `devctontainer.json` file and give you an option to run inside it. This requires you have `Docker` on your system, as well as some sort of Ubuntu distro (e.g. Windows: WSL Ubuntu).

The second option (creating virutal env locally) requires a few steps:
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

After installing and activating the virtual environment, the next step is installing dependencies:
1. For the training requirements, run the following command:
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements/requirements_train.txt
```

*Optional*: If you only want to run the models in inference, do:
```bash
pip install -r requirements/requirements_inference.txt
```

## Inference script
The inference script takes models from the default `models/` folder, and the images from the assigned folder, does the inferencing on each image, aggregating the results from all of the models present in the folder and outputs a `output_csv.csv` file as described in the competition specifications.

You can download the released models using the `download_models.py` script:
```bash
python models/download_models.py \
--repo-owner pipstur \
--repo-name MunjeLumenDS2025 --release-tag v2.0.0 \
--download-dir downloads/ \
--extract-dir models/
```

Example script run, assuming the images are inside the `data/test/` folder:
```bash
python inference/predict.py data/inference_test/ results/output_csv.csv
```

*Note*: If you do not have CUDA requirements highlighted in the `README_user.md`, you the `onnx-runtime` library will resort back to CPU inference. It will still run, but there will be warnings for each of the models, and the inference will be much slower.

## Model training
If the evaluator wishes to train the models, he can simply run the following command, everything is already set-up for him, if he followed the environment instructions closely, or is running inside a `devcontainer`.

He will need download the data first, using the following command:
```bash
python training/src/train_utils/dataset_preparation/get_datasets.py \
--output-dir data/get_data/
```
Then generate the training dataset:
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
And then run the training:
```bash
chmod +x schedule_training.sh
./schedule_training.sh
```
This results in models placed inside the `training/logs/train/runs/` folder, along with other data such as `tensorboard` files. If the evaluator wants to, he can inspect those with the following command:
```bash
tensorboard --logdir training/logs/train/runs
```
He can then follow the link (defaults to `http://localhost:6006/`) to check out each individual training run of the models.
