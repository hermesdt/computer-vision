import os
from pathlib import Path
import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from dataclasses import dataclass

class Trainer:

    def __init__(
        self,
        model: nn.Module,
        dataloder: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        snapshot_path: str,
        test_dataloader: DataLoader | None = None,
        lr_scheduler: LRScheduler | None = None,
        n_epochs: int = 10,
        epoch: int = 0,
        save_every: int = 1,
        grad_clip: float | None = None,
    ) -> None:
        self.model = model
        self.dataloader = dataloder
        self.test_dataloader = test_dataloader
        self.n_epochs = n_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = epoch
        self.save_every = save_every
        self.snapshot_path = snapshot_path
        self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler

        if os.path.exists(self.snapshot_path):
            self._load_snapshot(self.snapshot_path)
    
    def _load_snapshot(self, path: str) -> None:
        snapshot = torch.load(path)
        self.model.load_state_dict(snapshot['model_state_dict'])
        self.optimizer.load_state_dict(snapshot['optimizer_state_dict'])
        self.epoch = snapshot.get('epoch', 0)

    def _save_snapshot(self, path: str) -> None:
        snapshot = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
        }
        torch.save(snapshot, path)


    def step(self, batch, epoch: int) -> float:
        inputs, targets = batch
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()

        # Gradient clipping
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch)
        
        return loss.item()
        

    def train(self) -> None:
        self.model.train()

        for epoch in range(self.epoch, self.n_epochs):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(self.dataloader):
                epoch_loss += self.step(batch, epoch)
                break
            
            avg_loss = epoch_loss / len(self.dataloader)
            print(f"[Epoch={epoch}] Average Training Loss: {avg_loss}")

            self.evaluate()

            if (epoch + 1) % self.save_every == 0:
                self._save_snapshot(self.snapshot_path)
            
            self.epoch += 1
    
    def evaluate(self) -> None:
        self.model.eval()

        if self.test_dataloader is None:
            raise ValueError("Test dataloader is not provided.")
        
        with torch.no_grad():
            total_loss = 0.0
            for batch in self.test_dataloader:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

            avg_loss = total_loss / len(self.test_dataloader)
            print(f"[Epoch={self.epoch}] Average Test Loss: {avg_loss}")

