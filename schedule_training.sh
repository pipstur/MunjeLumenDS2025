# To get the models used in the final ensemble, run this script.
python training/src/train/train_kfold.py model=shufflenetv2 datamodule.data_dir=data/kfold_train/ model.freeze_layers=false datamodule.batch_size=256 trainer.max_epochs=100

python training/src/train/train_kfold.py model=mobilenetv3 datamodule.data_dir=data/kfold_train/ model.freeze_layers=false datamodule.batch_size=128 model.optimizer.lr=0.00001 trainer.max_epochs=100

# for some reason, squeezenet training slows down immencely when nearing the maximum VRAM capacity, so a reduced batch size is needed for the training to go smoothly. Convergence is still good
python training/src/train/train_kfold.py model=squeezenet1_1 datamodule.data_dir=data/kfold_train/ model.freeze_layers=false datamodule.batch_size=200 trainer.max_epochs=100 model.optimizer.lr=0.00001

python training/src/train/train_kfold.py model=efficientformer datamodule.data_dir=data/kfold_train/ model.freeze_layers=false trainer.max_epochs=100 datamodule.batch_size=110 model.optimizer.lr=0.0005 model.optimizer.weight_decay=0.00001
