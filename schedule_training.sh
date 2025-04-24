python training/src/train/train_kfold.py model=shufflenetv2 datamodule.data_dir=data/kfold_train_independent_test/ model.freeze_layers=false datamodule.batch_size=256 trainer.max_epochs=100

python training/src/train/train_kfold.py model=mobilenetv3 datamodule.data_dir=data/kfold_train_independent_test/ model.freeze_layers=false datamodule.batch_size=128 model.optimizer.lr=0.00001 trainer.max_epochs=100

python training/src/train/train_kfold.py model=squeezenet1_1 datamodule.data_dir=data/kfold_train_independent_test/ model.freeze_layers=false datamodule.batch_size=256 trainer.max_epochs=100 model.optimizer.lr=0.00001
