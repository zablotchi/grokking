import torch
import torch.nn as nn
import torch.nn.functional as F

from grokk_replica.transformer import Transformer
from grokk_replica.utils import causal_attn_mask, parameter_norm
from grokk_replica.losses import mse_loss_on_logits


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

    def get_loss(self, x, y, loss_key: str):
        predictions, attns = self(x)

        loss_dict: dict[str, torch.Tensor] = {
            "xent": F.cross_entropy(predictions[:, -1, :], y),
            "mse": mse_loss_on_logits(predictions[:, -1, :], y),
        }

        accuracy = (torch.argmax(predictions[:, -1, :], dim=-1) == y).float().mean()
        attn_entropies = sum(
            [
                -(attn * torch.log(attn + 1e-7)).sum(dim=-1).mean().item()
                for attn in attns
            ]
        ) / len(attns)
        param_norm = parameter_norm(self)

        return loss_dict[loss_key], {
            "loss": (loss_dict[loss_key].item(), x.shape[0]),
            "accuracy": (accuracy.item(), x.shape[0]),
            "attn_entropy": (
                attn_entropies,
                len(attns) * x.shape[0] * (x.shape[1] - 1),
            ),
            "param_norm": (param_norm, 1),
        } | {k: (v.item(), x.shape[0]) for k, v in loss_dict.items()}
