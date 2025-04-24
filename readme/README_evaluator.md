## Inference script
The inference script takes models from the default `models/` folder, and the images from the assigned folder, does the inferencing on each image, aggregating the results from all of the models present in the folder and outputs a `results.csv` file as described in the competition specifications.

You can download the released models using the `download_models.py` script:
```bash
python models/download_models.py \
--repo-owner pipstur \
--repo-name MunjeLumenDS2025 --release-tag v1.0.0 \
--download-dir downloads/ \
--extract-dir models/
```

Example script run, assuming the images are inside the `data/test/` folder:
```bash
python inference/inference.py \
--input-folder data/test/ \
--models-folder models/ \
--output-csv results/inference_results.csv \
--soft-vote
```

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
