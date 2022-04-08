import dataclasses

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset


@dataclasses.dataclass
class InfiniteDMConfig:
    batch_size: int = 512
    n_workers: int = 16


class InfiniteDM(pl.LightningDataModule):
    def __init__(self, ds: Dataset, cfg: InfiniteDMConfig):
        super().__init__()
        self.ds = ds
        self.cfg = cfg

    def train_dataloader(self):
        return DataLoader(
            self.ds,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.n_workers,
            shuffle=False,
        )


@dataclasses.dataclass
class OneHotDSConfig:
    n_classes: int


class OneHotDS(IterableDataset):
    def __init__(self, cfg: OneHotDSConfig):
        super().__init__()
        self.cfg = cfg

    def __iter__(self):
        while True:
            y_int = torch.randint(low=0, high=self.cfg.n_classes, size=())
            x: torch.Tensor = F.one_hot(y_int, num_classes=self.cfg.n_classes)
            yield x.float(), y_int / self.cfg.n_classes
