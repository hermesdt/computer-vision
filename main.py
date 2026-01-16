from importlib import import_module
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from timm.data.dataset import ImageDataset
from src.trainer import Trainer
from src.utils import dataset as dataset_utils
import hydra
from omegaconf import DictConfig, OmegaConf
import timm
import timm.data.dataset_factory
import timm.data.loader

def _import_class(full_class_string: str) -> type:
    module_name, class_name = full_class_string.rsplit(".", 1)
    module = import_module(module_name)
    cls = getattr(module, class_name)
    return cls

def create_model(cfg: DictConfig) -> nn.Module:
    return timm.create_model(cfg["name"], **cfg["params"])

# def create_transforms(cfg: DictConfig) -> list[nn.Module]:
#     transforms = []
#     for name, params in cfg.items():
#         transform_class = _import_class(name)
#         transform = transform_class(**params)
#         transforms.append(transform)
#     return transforms


def create_dataloader(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    dataset_path = dataset_utils.download(cfg["name"], cfg["data_folder"])
    ds = timm.data.dataset_factory.create_dataset(
        root=dataset_path,
        name="folder",
        split="train",
        is_training=True,
    )

    dl = timm.data.loader.create_loader(
        ds,
        input_size=(3, 224, 224),
        batch_size=cfg["batch_size"],
        device=torch.device("mpx" if torch.cuda.is_available() else "cpu"),
    )

    return dl, dl

def create_criterion(cfg: DictConfig) -> nn.Module:
    criterion_class = _import_class(cfg["name"])
    criterion = criterion_class(**cfg["params"])
    return criterion

def create_optimizer(model: nn.Module, cfg: DictConfig) -> optim.Optimizer:
    optimizer_class = _import_class(cfg["name"])
    optimizer = optimizer_class(model.parameters(), **cfg["params"])
    return optimizer

def create_trainer(
    cfg: DictConfig,
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader | None,
    criterion: nn.Module,
    optimizer: optim.Optimizer
) -> Trainer:
    trainer = Trainer(
        model=model,
        dataloder=train_dataloader,
        test_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        snapshot_path=cfg["snapshot_path"],
        n_epochs=cfg["n_epochs"],
        save_every=cfg["save_every"],
        grad_clip=cfg.get("grad_clip", None),
    )
    return trainer

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    model = create_model(cfg["model"])
    train_dataloader, val_dataloader = create_dataloader(cfg["dataset"])
    
    criterion = create_criterion(cfg["training"]["loss"])
    optimizer = create_optimizer(model, cfg["training"]["optimizer"])

    trainer = create_trainer(
        cfg["training"],
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
