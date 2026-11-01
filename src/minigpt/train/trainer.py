from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from minigpt.train.checkpoint import save_checkpoint
from minigpt.utils import count_parameters, ensure_dir, save_json


def _cosine_lr(step: int, warmup: int, total: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * step / max(1, warmup)
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader | None,
        cfg: dict[str, Any],
        device: torch.device,
        run_dir: str | Path,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.cfg = cfg
        self.device = device
        self.run_dir = ensure_dir(run_dir)
        tcfg = cfg["train"]
        decay, nodecay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or name.endswith("bias") or "ln" in name.lower() or "norm" in name.lower():
                nodecay.append(param)
            else:
                decay.append(param)
        self.optimizer = AdamW(
            [
                {"params": decay, "weight_decay": float(tcfg.get("weight_decay", 0.1))},
                {"params": nodecay, "weight_decay": 0.0},
            ],
            lr=float(tcfg["lr"]),
            betas=(0.9, 0.95),
        )
        self.use_amp = bool(tcfg.get("amp", False)) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.history: list[dict[str, float]] = []
        self.best_val = float("inf")
        self.global_step = 0

    def train(self) -> dict[str, Any]:
        tcfg = self.cfg["train"]
        epochs = int(tcfg["epochs"])
        accum = int(tcfg.get("grad_accum", 1))
        clip = float(tcfg.get("grad_clip", 1.0))
        warmup = int(tcfg.get("warmup_steps", 0))
        max_steps = tcfg.get("max_steps")
        steps_per_epoch = math.ceil(len(self.train_loader) / accum)
        total_steps = int(max_steps) if max_steps else steps_per_epoch * epochs
        lr = float(tcfg["lr"])
        min_lr = float(tcfg.get("min_lr", lr * 0.1))
        n_params = count_parameters(self.model)
        print(f"Trainable parameters: {n_params:,}")
        save_json(
            {"n_params": n_params, "device": str(self.device), "model": self.cfg.get("model", {})},
            self.run_dir / "model_info.json",
        )

        for epoch in range(1, epochs + 1):
            self.model.train()
            running = 0.0
            n_batches = 0
            self.optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(self.train_loader, desc=f"epoch {epoch}/{epochs}")
            for i, batch in enumerate(pbar, start=1):
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    out = self.model(input_ids, labels=labels)
                    loss = out["loss"] / accum
                self.scaler.scale(loss).backward()
                running += float(out["loss"].detach().cpu())
                n_batches += 1
                if i % accum == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    cur_lr = _cosine_lr(self.global_step, warmup, total_steps, lr, min_lr)
                    for group in self.optimizer.param_groups:
                        group["lr"] = cur_lr
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    pbar.set_postfix(loss=f"{running / n_batches:.4f}", lr=f"{cur_lr:.2e}")
                    if max_steps and self.global_step >= int(max_steps):
                        break
            train_loss = running / max(n_batches, 1)
            val_loss, val_ppl = self.evaluate()
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "step": self.global_step,
                "n_params": n_params,
            }
            self.history.append(row)
            save_json(self.history, self.run_dir / "history.json")
            save_checkpoint(
                self.run_dir / "last.pt",
                self.model,
                self.optimizer,
                extra={"epoch": epoch, "history": self.history},
            )
            if val_loss < self.best_val:
                self.best_val = val_loss
                save_checkpoint(
                    self.run_dir / "best.pt",
                    self.model,
                    extra={"epoch": epoch, "val_loss": val_loss, "val_ppl": val_ppl},
                )
            print(
                f"epoch {epoch}: train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}"
            )
            if max_steps and self.global_step >= int(max_steps):
                break
        return {"best_val_loss": self.best_val, "history": self.history, "n_params": n_params}

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        if self.valid_loader is None:
            return float("nan"), float("nan")
        self.model.eval()
        total = 0.0
        n = 0
        for batch in self.valid_loader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                out = self.model(input_ids, labels=labels)
            total += float(out["loss"].detach().cpu()) * input_ids.size(0)
            n += input_ids.size(0)
        mean_loss = total / max(n, 1)
        ppl = math.exp(min(mean_loss, 20))
        return mean_loss, ppl
