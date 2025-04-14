from pathlib import Path
from typing import Any, Dict, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


class MelanomaDataModule(LightningDataModule):
    """Melanoma implementation of LightningDataModule.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "",
        batch_size: int = 64,
        imbalanced_sampling: bool = False,
        num_workers: int = 0,
        tile_size: int = [224, 224],
        pin_memory: bool = False,
        grayscale: bool = False,
        dirs: str = ["train", "val", "test"],
        train_da: bool = False,
        val_da: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.dirs = dirs
        self.imbalanced_sampling = imbalanced_sampling
        self.class_names = ["benign", "malignant"]

        train_trans = []
        val_trans = []
        test_trans = []

        trans = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomVerticalFlip(),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            # transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        ]
        trans_for_all = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        if train_da:
            train_trans += trans
        if val_da:
            val_trans += trans

        train_trans += trans_for_all
        val_trans += trans_for_all
        test_trans += trans_for_all

        self.train_transforms = transforms.Compose(train_trans)
        self.val_transforms = transforms.Compose(val_trans)
        self.test_transforms = transforms.Compose(test_trans)

    def setup(self, stage: Optional[str] = None):
        self.data_train = ImageFolder(
            Path(self.data_dir) / self.dirs[0], transform=self.train_transforms
        )
        self.data_val = ImageFolder(
            Path(self.data_dir) / self.dirs[1], transform=self.val_transforms
        )
        self.data_test = ImageFolder(
            Path(self.data_dir) / self.dirs[2], transform=self.test_transforms
        )

    def train_dataloader(self):
        if self.imbalanced_sampling:
            return DataLoader(
                dataset=self.data_train,
                sampler=ImbalancedDatasetSampler(self.data_train),
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                persistent_workers=True,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            )
        else:
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                persistent_workers=True,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
            )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "default.yaml")
    cfg.data_dir = str(root / "datasets")
    _ = hydra.utils.instantiate(cfg)
