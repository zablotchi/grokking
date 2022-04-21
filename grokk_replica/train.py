import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm

from grokk_replica.datasets import AbstractDataset
from grokk_replica.load_objs import load_item
from grokk_replica.utils import combine_logs


class GroupDataset(IterableDataset):
    def __init__(self, dataset: AbstractDataset, split: str):
        super(GroupDataset, self).__init__()
        assert split in {"train", "val"}
        self.dataset = dataset
        self.split = split
        self.fetch_f = None
        if self.split == "train":
            self.fetch_f = self.dataset.fetch_train_example
        elif self.split == "val":
            self.fetch_f = self.dataset.fetch_val_example
        else:
            raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        x, y, _ = self.fetch_f()
        return torch.tensor(x), torch.tensor(y)


def get_eval_logs(
    model: nn.Module,
    dl: DataLoader,
    n_batches: int,
    device: torch.device,
):
    with torch.no_grad():
        all_logs = []
        for i, (x, y) in enumerate(dl):
            if i >= n_batches:
                break
            _, logs = model.get_loss(x.to(device), y.to(device))
            all_logs.append(logs)

    return all_logs


def train(config):
    print("using config:", config)
    train_cfg = config["train"]
    wandb_cfg = config["wandb"]
    if wandb_cfg["use_wandb"]:
        wandb.init(
            entity=wandb_cfg["wandb_entity"],
            project=wandb_cfg["wandb_project"],
            tags=wandb_cfg["wandb_tags"],
            config=config,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_item(config["dataset"])
    train_data = GroupDataset(dataset, "train")
    val_data = GroupDataset(dataset, "val")
    model = load_item(config["model"], dataset.n_vocab, dataset.n_out, device)
    model.train()
    train_dataloader = DataLoader(
        train_data, num_workers=train_cfg["num_workers"], batch_size=train_cfg["bsize"]
    )
    val_dataloader = DataLoader(
        val_data, num_workers=train_cfg["num_workers"], batch_size=train_cfg["bsize"]
    )
    optim = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        betas=train_cfg["betas"],
    )
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, 'min', patience=50, verbose=True
    )
    # torch.optim.lr_scheduler.LambdaLR(
    #     optim, lr_lambda=lambda s: min(s / train_cfg["warmup_steps"], 1)
    # )
    step = 0
    pbar = tqdm(train_dataloader)
    for x, y in pbar:
        loss, _ = model.get_loss(x.to(device), y.to(device))
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schedule.step(loss.detach())
        if (step + 1) % train_cfg["eval_every"] == 0:
            model.eval()

            train_logs = get_eval_logs(
                model=model,
                dl=train_dataloader,
                n_batches=train_cfg["eval_batches"],
                device=device,
            )
            val_logs = get_eval_logs(
                model=model,
                dl=val_dataloader,
                n_batches=train_cfg["eval_batches"],
                device=device,
            )

            out_log = {
                "val": combine_logs(val_logs),
                "train": combine_logs(train_logs),
                "step": (step + 1),
                # "lr": float(lr_schedule.get_last_lr()[0]),
            }
            if wandb_cfg["use_wandb"]:
                wandb.log(out_log)
            model.train()

            pbar.set_description(f"train.rmse: {out_log['train']['rmse']: .6e}")
        step += 1
        if train_cfg["max_steps"] is not None and step >= train_cfg["max_steps"]:
            break


@hydra.main(config_path="../config", config_name="train_grokk")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
