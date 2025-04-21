## Inference script
The inference script takes models from the default `models/` folder, and the images from the assigned folder, does the inferencing on each image, aggregating the results from all of the models present in the folder and outputs a `results.csv` file as described in the competition specifications.

### Download the models
You can download the released models using the `download_models.py` script:

```bash
python models/download_models.py \
--repo-owner pipstur \
--repo-name MunjeLumenDS2025 --release-tag v1.0.0 \
--download-dir downloads/ \
--extract-dir models/
```

### Running the inference script
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

```bash
chmod +x schedule_training.sh
./schedule_training.sh
```
This results in models placed inside the `training/logs/train/runs/` folder, along with other data such as `tensorboard` files. If the evaluator wants to, he can inspect those with the following command:
```bash
tensorboard --logdir training/logs/train/runs
```
He can then follow the link (defaults to `http://localhost:6006/`) to check out each individual training run of the models.
