"""TUNI Training Script — ANIMA Standards Compliant.

Config-driven, checkpoint-managed, early-stopping, NaN-safe.
Usage:
    CUDA_VISIBLE_DEVICES=0 python -m def_tuni.train --config configs/fmb_paper.toml
    CUDA_VISIBLE_DEVICES=0 python -m def_tuni.train --config configs/debug.toml --max-steps 10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from .model import TUNIModel
from .data import get_dataset, DATASET_DEFAULTS
from .metrics import RunningScore


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * scale)

    def state_dict(self):
        return {'current_step': self.current_step}

    def load_state_dict(self, state):
        self.current_step = state['current_step']


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    def __init__(self, save_dir: Path, keep_top_k: int = 2, metric: str = 'val_loss', mode: str = 'min'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.metric = metric
        self.mode = mode
        self.history: list[tuple[float, Path]] = []

    def save(self, state: dict, metric_value: float, step: int):
        path = self.save_dir / f'checkpoint_step{step:06d}.pth'
        torch.save(state, path)
        self.history.append((metric_value, path))
        self.history.sort(key=lambda x: x[0], reverse=(self.mode == 'max'))

        while len(self.history) > self.keep_top_k:
            _, old_path = self.history.pop()
            old_path.unlink(missing_ok=True)

        best_path = self.save_dir / 'best.pth'
        if self.history:
            shutil.copy2(self.history[0][1], best_path)


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0

    def step(self, metric: float) -> bool:
        improved = (metric < self.best - self.min_delta) if self.mode == 'min' \
            else (metric > self.best + self.min_delta)
        if improved:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def group_weight_decay(model: nn.Module):
    decays, no_decays = [], []
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            decays.append(m.weight)
            if m.bias is not None:
                no_decays.append(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            if m.weight is not None:
                no_decays.append(m.weight)
            if m.bias is not None:
                no_decays.append(m.bias)
    return [dict(params=decays), dict(params=no_decays, weight_decay=0.0)]


def train(cfg: dict, resume: str | None = None, max_steps: int | None = None):
    # Unpack config
    tcfg = cfg["training"]
    dcfg = cfg["data"]
    ccfg = cfg["checkpoint"]

    device_index = int(os.environ.get('ANIMA_CUDA_DEVICE', '0'))
    device = torch.device(f'cuda:{device_index}' if torch.cuda.is_available() else 'cpu')

    # Seed
    seed = tcfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Dataset
    dataset_name = dcfg["dataset"]
    defaults = DATASET_DEFAULTS.get(dataset_name.lower(), {})
    n_classes = defaults.get("n_classes", dcfg.get("n_classes", 9))

    train_set = get_dataset(dataset_name, dcfg["train_path"], mode='train')
    # Try to get validation set
    try:
        val_set = get_dataset(dataset_name, dcfg["val_path"], mode='test')
    except Exception:
        val_set = get_dataset(dataset_name, dcfg["val_path"], mode='val')

    batch_size = tcfg["batch_size"]
    num_workers = dcfg.get("num_workers", 4)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # Model
    model_cfg = cfg.get("model", {})
    variant = model_cfg.get("variant", "384_2242")
    pretrained = model_cfg.get("pretrained", None)
    model = TUNIModel(
        variant=variant,
        n_classes=n_classes,
        drop_path_rate=model_cfg.get("drop_path_rate", 0.1),
        pretrained=pretrained,
    ).to(device)

    # Load full pretrained checkpoint (encoder + decoder) if specified
    pretrained_ckpt = model_cfg.get("pretrained_checkpoint", None)
    if pretrained_ckpt:
        model.load_checkpoint(pretrained_ckpt)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[MODEL] TUNI variant={variant}, {n_params:.1f}M params, n_classes={n_classes}")

    # Class weight
    if hasattr(train_set, 'class_weight'):
        class_weight = train_set.class_weight.to(device)
    else:
        class_weight = None

    # Ignore index
    ignore_index = dcfg.get("id_unlabel", -100)
    criterion = nn.CrossEntropyLoss(weight=class_weight, ignore_index=ignore_index)

    # Optimizer
    lr = tcfg["learning_rate"]
    wd = tcfg.get("weight_decay", 0.01)
    param_groups = group_weight_decay(model)
    param_groups[0]['weight_decay'] = wd
    optimizer = torch.optim.AdamW(param_groups, lr=lr)

    # Scheduler
    epochs = tcfg["epochs"]
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * tcfg.get("warmup_ratio", 0.05))
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps,
                                       min_lr=tcfg.get("min_lr", 1e-7))

    # Precision
    precision = tcfg.get("precision", "fp16")
    use_amp = precision in ("fp16", "bf16") and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    # Gradient clipping
    max_grad_norm = tcfg.get("max_grad_norm", 1.0)

    # Checkpoint manager
    ckpt_dir = Path(ccfg.get("save_dir", "/mnt/artifacts-datai/checkpoints/DEF-tuni"))
    ckpt_manager = CheckpointManager(
        ckpt_dir, keep_top_k=ccfg.get("keep_top_k", 2),
        metric=ccfg.get("metric", "mIoU"), mode=ccfg.get("mode", "max"),
    )
    save_every = ccfg.get("save_every_n_steps", 500)

    # Early stopping
    es_cfg = cfg.get("early_stopping", {})
    early_stop = EarlyStopping(
        patience=es_cfg.get("patience", 20),
        min_delta=es_cfg.get("min_delta", 0.001),
        mode="max",  # higher mIoU = better
    ) if es_cfg.get("enabled", True) else None

    # Resume
    start_epoch = 0
    global_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('step', 0)
        print(f"[RESUME] from epoch {start_epoch}, step {global_step}")

    # Print config summary
    print(f"[CONFIG] {cfg.get('_source', 'inline')}")
    print(f"[BATCH] batch_size={batch_size}")
    print(f"[GPU] {torch.cuda.get_device_name(device)}")
    print(f"[DATA] train={len(train_set)}, val={len(val_set)}")
    print(f"[TRAIN] {epochs} epochs, lr={lr}, optimizer=AdamW, precision={precision}")
    print(f"[CKPT] save every {save_every} steps, keep best {ccfg.get('keep_top_k', 2)}")
    print(f"[SCHEDULER] warmup={warmup_steps} steps, total={total_steps}")

    # Log file
    log_dir = Path(ccfg.get("log_dir", "/mnt/artifacts-datai/logs/DEF-tuni"))
    log_dir.mkdir(parents=True, exist_ok=True)

    best_miou = 0.0

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        for i, sample in enumerate(train_loader):
            if max_steps is not None and global_step >= max_steps:
                print(f"[STOP] max_steps={max_steps} reached")
                return

            rgb = sample['image'].to(device, non_blocking=True)
            thermal = sample['thermal'].to(device, non_blocking=True)
            label = sample['label'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                pred = model(rgb, thermal)
                pred = F.interpolate(pred, size=label.shape[1:], mode='bilinear', align_corners=False)
                loss = criterion(pred, label)

            if torch.isnan(loss):
                print(f'[FATAL] Loss is NaN at step {global_step} — stopping')
                return

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            epoch_steps += 1

            if global_step % 50 == 0:
                lr_now = optimizer.param_groups[0]['lr']
                throughput = batch_size * 50 / (time.time() - t0 + 1e-6)
                print(f"  step {global_step} | loss={loss.item():.4f} | lr={lr_now:.2e} | {throughput:.1f} img/s")
                t0 = time.time()

            # Step-based checkpoint
            if global_step % save_every == 0:
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'step': global_step,
                    'config': cfg,
                }
                ckpt_manager.save(state, best_miou, global_step)
                print(f"  [CKPT] saved at step {global_step}")

        avg_loss = epoch_loss / max(epoch_steps, 1)

        # Validation
        model.eval()
        scorer = RunningScore(n_classes, ignore_index=ignore_index if ignore_index >= 0 else None)
        val_loss_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for sample in val_loader:
                rgb = sample['image'].to(device, non_blocking=True)
                thermal = sample['thermal'].to(device, non_blocking=True)
                label = sample['label'].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    pred = model(rgb, thermal)
                    pred = F.interpolate(pred, size=label.shape[1:], mode='bilinear', align_corners=False)
                    vloss = criterion(pred, label)

                val_loss_sum += vloss.item()
                val_steps += 1

                pred_np = pred.argmax(dim=1).cpu().numpy()
                label_np = label.cpu().numpy()
                scorer.update(label_np, pred_np)

        scores = scorer.get_scores()
        miou = scores['mIoU']
        val_loss = val_loss_sum / max(val_steps, 1)

        print(f"[EPOCH {epoch+1}/{epochs}] train_loss={avg_loss:.4f} | "
              f"val_loss={val_loss:.4f} | mIoU={miou*100:.2f}% | "
              f"pixel_acc={scores['pixel_acc']*100:.2f}%")

        # Save best checkpoint
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'step': global_step,
            'metrics': scores,
            'config': cfg,
        }
        ckpt_manager.save(state, miou, global_step)

        if miou > best_miou:
            best_miou = miou
            print(f"  [BEST] new best mIoU={best_miou*100:.2f}%")

        # Early stopping
        if early_stop and early_stop.step(miou):
            print(f"[EARLY STOP] No improvement for {early_stop.patience} epochs")
            break

    print(f"\n[DONE] Training complete. Best mIoU={best_miou*100:.2f}%")
    print(f"[CKPT] Best model at {ckpt_dir / 'best.pth'}")


def main():
    parser = argparse.ArgumentParser(description="TUNI Training")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps (for smoke test)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["_source"] = args.config
    train(cfg, resume=args.resume, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
