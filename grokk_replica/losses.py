import torch
import einops


def mse_loss_on_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    logits: (batch, num_classes)
    targets: (batch)

    Averages across the batch.
    """
    _, num_classes = logits.shape

    diffs = (
        einops.rearrange(
            torch.arange(num_classes, device=targets.device),
            "c -> 1 c",
        )
        - einops.rearrange(
            targets,
            "b -> b 1",
        )
    ) / (num_classes - 2)
    # We divide by (num_classes - 2) to make the output lie in [0, 1].

    mse_loss = torch.einsum(
        "bi, bi -> b",
        diffs**2,
        logits.softmax(dim=-1),
    )
    return mse_loss.mean()
