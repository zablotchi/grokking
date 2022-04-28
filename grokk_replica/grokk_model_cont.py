import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from grokk_replica.transformer import Transformer
from grokk_replica.utils import causal_attn_mask, parameter_norm


class GrokkModelContOut(nn.Module):
    def __init__(self, transformer_config, vocab_size, output_size, device):
        super().__init__()
        hidden_dim = transformer_config["hidden_dim"]

        self.transformer = Transformer(
            **transformer_config, vocab_size=vocab_size, output_size=hidden_dim
        )
        self.cont_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.device = device

    def forward(self, x):
        attn_mask = (
            causal_attn_mask(x.shape[1])
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1)
            .to(self.device)
        )
        transformer_embeddings, attns, _ = self.transformer(x, attn_mask)
        predictions = einops.rearrange(
            self.cont_head(transformer_embeddings),
            "... 1 -> ...",
        )
        return predictions, attns

    def get_loss(self, x, y):
        predictions, attns = self(x)
        mse = F.mse_loss(predictions[:, -1], y)
        attn_entropies = sum(
            [
                -(attn * torch.log(attn + 1e-7)).sum(dim=-1).mean().item()
                for attn in attns
            ]
        ) / len(attns)
        param_norm = parameter_norm(self)
        return mse, {
            "loss": (mse.item(), x.shape[0]),
            "mse": (mse.item(), x.shape[0]),
            "rmse": (mse.sqrt().item(), x.shape[0]),
            "attn_entropy": (
                attn_entropies,
                len(attns) * x.shape[0] * (x.shape[1] - 1),
            ),
            "param_norm": (param_norm, 1),
        }
