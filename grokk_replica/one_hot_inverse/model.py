import dataclasses

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class OneHotFCNetConfig:
    n_classes: int
    hidden_width: int
    depth: int

    opt_type: str = "adam"  # or adamw

    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.98)


class OneHotFCNet(pl.LightningModule):
    def __init__(self, cfg: OneHotFCNetConfig):
        super().__init__()
        self.cfg = cfg

        layers: list[nn.Module] = [
            nn.Linear(cfg.n_classes, cfg.hidden_width),
            nn.ReLU(),
        ]
        for _ in range(cfg.depth - 2):
            layers.append(nn.Linear(cfg.hidden_width, cfg.hidden_width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(cfg.hidden_width, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(x)

    def training_step(self, batch, *_, **__):
        xs, ys = batch
        logits: torch.Tensor = self(xs)
        mse = F.mse_loss(logits.squeeze(-1), ys)
        self.log("mse", mse)

        return mse

    def configure_optimizers(self):
        if self.cfg.opt_type == "adam":
            opt = torch.optim.Adam(
                self.parameters(),
                lr=self.cfg.learning_rate,
            )
        elif self.cfg.opt_type == "adamw":
            opt = torch.optim.AdamW(
                self.parameters(),
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay,
                betas=self.cfg.betas,
            )
        else:
            raise NotImplementedError

        return [opt], []
