import einops
import torch
import torch.nn.functional as F
from torch import nn

from grokk_replica.transformer import Transformer
from grokk_replica.utils import causal_attn_mask, parameter_norm


class GrokkModel(nn.Module):
    def __init__(self, transformer_config, vocab_size, output_size, device):
        super(GrokkModel, self).__init__()
        self.transformer = Transformer(
            **transformer_config, vocab_size=vocab_size, output_size=output_size
        )
        self.device = device

    def forward(self, x):
        attn_mask = (
            causal_attn_mask(x.shape[1])
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1)
            .to(self.device)
        )
        predictions, attns, _ = self.transformer(x, attn_mask)
        return predictions, attns

    def get_loss(self, x, y):
        predictions, attns = self(x)
        # print(torch.argmax(predictions[:, -1, :], dim=-1), x[:, -1])
        loss = F.cross_entropy(predictions[:, -1, :], y)
        accuracy = (torch.argmax(predictions[:, -1, :], dim=-1) == y).float().mean()
        attn_entropies = sum(
            [
                -(attn * torch.log(attn + 1e-7)).sum(dim=-1).mean().item()
                for attn in attns
            ]
        ) / len(attns)
        param_norm = parameter_norm(self)
        return loss, {
            "loss": (loss.item(), x.shape[0]),
            "accuracy": (accuracy.item(), x.shape[0]),
            "attn_entropy": (
                attn_entropies,
                len(attns) * x.shape[0] * (x.shape[1] - 1),
            ),
            "param_norm": (param_norm, 1),
        }


class GrokkModelFC(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        output_size: int,
        num_layers: int,
        device: str,
    ):
        super().__init__()
        assert num_layers >= 2

        self.vocab_size = vocab_size
        self.device = device

        self.net = nn.Sequential(
            nn.Sequential(
                nn.Linear(in_features=vocab_size * 4, out_features=hidden_dim),
                nn.GELU(),
            ),
            *(
                nn.Sequential(
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                    nn.GELU(),
                )
                for _ in range(num_layers - 2)
            ),
            nn.Linear(in_features=hidden_dim, out_features=output_size)
        )

    def forward(self, x: torch.Tensor):
        x_one_hot = F.one_hot(
            x.to(self.device),
            num_classes=self.vocab_size,
        ).float()
        return self.net(
            einops.rearrange(
                x_one_hot,
                "... seq vocab_size -> ... (seq vocab_size)",
            )
        )

    def get_loss(self, x: torch.Tensor, y: torch.Tensor):
        predictions = self(x)
        loss = F.cross_entropy(predictions, y)
        accuracy = (torch.argmax(predictions, dim=-1) == y).float().mean()
        param_norm = parameter_norm(self)
        return loss, {
            "loss": (loss.item(), x.shape[0]),
            "accuracy": (accuracy.item(), x.shape[0]),
            "param_norm": (param_norm, 1),
        }
