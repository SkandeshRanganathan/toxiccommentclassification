"""Research-grade training loop with AMP, resumption, and best-model selection."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from training.early_stopping import EarlyStopping
from training.evaluator import evaluate_model, model_inputs
from training.losses import MultiLabelObjective
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logger import SummaryWriterAdapter


class Trainer:
    def __init__(self, model: torch.nn.Module, train_loader: DataLoader, validation_loader: DataLoader, objective: MultiLabelObjective, config: Any, device_policy: Any, logger: Any, label_names: Sequence[str], run_dir: str | Path) -> None:
        self.model = model.to(device_policy.device)
        self.train_loader, self.validation_loader = train_loader, validation_loader
        self.objective, self.config, self.policy = objective.to(device_policy.device), config, device_policy
        self.logger, self.label_names = logger, tuple(label_names)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriterAdapter(self.run_dir / "tensorboard")
        self.optimizer = self._build_optimizer()
        accumulation = int(config.training.gradient_accumulation_steps)
        updates_per_epoch = math.ceil(len(train_loader) / accumulation)
        total_steps = updates_per_epoch * int(config.training.epochs)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, int(total_steps * float(config.training.warmup_ratio)), total_steps)
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=device_policy.amp_enabled)
        except (AttributeError, TypeError):  # PyTorch 2.1 compatibility.
            self.scaler = torch.cuda.amp.GradScaler(enabled=device_policy.amp_enabled)
        self.early_stopping = EarlyStopping(int(config.training.early_stopping_patience), float(config.training.early_stopping_min_delta), "max" if config.training.monitor != "loss" else "min")
        self.start_epoch, self.global_step, self.history = 0, 0, []
        if config.training.resume_from:
            self._resume(config.training.resume_from)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        encoder, remainder = [], []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                (encoder if name.startswith("encoder.") else remainder).append((name, parameter))
        groups = []
        for parameters, learning_rate in ((encoder, float(self.config.training.encoder_learning_rate)), (remainder, float(self.config.training.learning_rate))):
            for decay in (True, False):
                selected = [parameter for name, parameter in parameters if all(token not in name for token in no_decay) == decay]
                if selected:
                    groups.append({"params": selected, "lr": learning_rate, "weight_decay": float(self.config.training.weight_decay) if decay else 0.0})
        return torch.optim.AdamW(groups, betas=(float(self.config.training.adam_beta1), float(self.config.training.adam_beta2)), eps=float(self.config.training.adam_epsilon))

    def _resume(self, checkpoint: str | Path) -> None:
        payload = load_checkpoint(checkpoint, self.model, self.optimizer, self.scheduler, self.scaler, self.policy.device, restore_rng=True)
        self.start_epoch, self.global_step = int(payload["epoch"]) + 1, int(payload["global_step"])
        self.early_stopping.load_state_dict(payload.get("early_stopping", {}))
        self.logger.info("Resumed %s at epoch %d", checkpoint, self.start_epoch)

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        accumulation = int(self.config.training.gradient_accumulation_steps)
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", leave=False)
        for batch_index, batch in enumerate(progress):
            labels = batch["labels"].to(self.policy.device, non_blocking=True)
            with self.policy.autocast():
                output = self.model(**model_inputs(batch, self.policy.device))
                raw_loss = self.objective(output.logits, labels, output.auxiliary_loss)
                loss = raw_loss / accumulation
            self.scaler.scale(loss).backward()
            total_loss += float(raw_loss.item()) * labels.shape[0]
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.training.max_grad_norm))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                self.writer.add_scalar("train/step_loss", float(raw_loss.item()), self.global_step)
            progress.set_postfix(loss=f"{raw_loss.item():.4f}")
        return total_loss / len(self.train_loader.dataset)

    def fit(self) -> dict[str, float]:
        best_metrics: dict[str, float] = {}
        try:
            for epoch in range(self.start_epoch, int(self.config.training.epochs)):
                started = time.time()
                train_loss = self._train_epoch(epoch)
                result = evaluate_model(self.model, self.validation_loader, self.objective, self.policy.device, self.label_names, amp_context=self.policy.autocast)
                row = {"epoch": float(epoch + 1), "train_loss": train_loss, **result.metrics, "seconds": time.time() - started}
                self.history.append(row)
                pd.DataFrame(self.history).to_csv(self.run_dir / "history.csv", index=False)
                for key, value in row.items():
                    self.writer.add_scalar(f"epoch/{key}", value, epoch + 1)
                monitor = float(result.metrics[self.config.training.monitor])
                improved, stop = self.early_stopping.update(monitor)
                save_checkpoint(self.run_dir / "last.pt", self.model, self.optimizer, self.scheduler, self.scaler, epoch, self.global_step, result.metrics, self.config, self.early_stopping.state_dict())
                if improved:
                    best_metrics = result.metrics
                    save_checkpoint(self.run_dir / "best.pt", self.model, self.optimizer, self.scheduler, self.scaler, epoch, self.global_step, result.metrics, self.config, self.early_stopping.state_dict())
                self.logger.info("Epoch %d train_loss=%.5f %s=%.5f%s", epoch + 1, train_loss, self.config.training.monitor, monitor, " [best]" if improved else "")
                if stop:
                    self.logger.info("Early stopping after epoch %d", epoch + 1)
                    break
        finally:
            self.writer.close()
        return best_metrics
