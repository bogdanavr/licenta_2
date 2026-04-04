import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import csv
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from train.config import EXPORT_DIR, IMG_SIZE, LOG_DIR, NUM_CLASSES, SEED
from train.dataset import EmotionDataset
from train.model import build_model

BATCH_SIZE = 32
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
PATIENCE = 5

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def build_train_transform():
    return T.Compose(
        [
            T.RandomResizedCrop(
                IMG_SIZE,
                scale=(0.75, 1.0),
                interpolation=InterpolationMode.BILINEAR,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(12),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
            T.RandomErasing(p=0.15, scale=(0.02, 0.12)),
        ]
    )



def build_eval_transform():
    return T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )



def build_dataloaders(device):
    train_ds = EmotionDataset(split="train", transform=build_train_transform())
    val_ds = EmotionDataset(split="val", transform=build_eval_transform())

    common_loader_args = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": NUM_WORKERS > 0,
    }

    train_loader = DataLoader(train_ds, shuffle=True, **common_loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader_args)
    return train_ds, val_ds, train_loader, val_loader



def compute_class_weights(targets, device):
    counts = np.bincount(np.array(targets), minlength=NUM_CLASSES)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)



def get_optimizer(model):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=LR,
    )



def run_epoch(model, loader, criterion, optimizer, device, scaler=None, epoch=None, phase="train"):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    use_amp = device.type == "cuda"

    progress = tqdm(loader, desc=f"Ep {epoch:02d} [{phase}]", leave=False)

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        running_loss = total_loss / total_samples
        running_acc = total_correct / total_samples
        progress.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

    return total_loss / total_samples, total_correct / total_samples



def save_history(history):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    history_json = LOG_DIR / "train_history.json"
    with history_json.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, ensure_ascii=False)

    history_csv = LOG_DIR / "train_log.csv"
    with history_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"],
        )
        writer.writeheader()
        writer.writerows(history)



def main():
    seed_everything(SEED)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, _, train_loader, val_loader = build_dataloaders(device)

    model = build_model(pretrained=True).to(device)
    class_weights = compute_class_weights(train_ds.targets, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = get_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val_acc = 0.0
    epochs_without_improvement = 0
    history = []

    print(
        f"Antrenare EfficientNetV2-S | device={device} | batch_size={BATCH_SIZE} | epochs={EPOCHS}"
    )

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            epoch=epoch,
            phase="train",
        )
        with torch.no_grad():
            val_loss, val_acc = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                scaler=None,
                epoch=epoch,
                phase="val",
            )

        scheduler.step(val_acc)
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "lr": round(current_lr, 8),
            "seconds": round(elapsed, 2),
        }
        history.append(row)

        print(
            f"Epoca {epoch:02d} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f} "
            f"| train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={current_lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "img_size": IMG_SIZE,
                },
                EXPORT_DIR / "best_model.pth",
            )
            print(f"Checkpoint nou salvat cu val_acc={best_val_acc:.4f}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping activat.")
            break

    save_history(history)
    print(f"Antrenarea s-a incheiat. Cel mai bun val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
