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
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from train.config import (
    EXPORT_DIR,
    IMAGE_MEAN,
    IMAGE_STD,
    IMG_SIZE,
    LOG_DIR,
    MODEL_DISPLAY_NAME,
    MODEL_NAME,
    SAVED_IMG_SIZE,
    SEED,
    TARGET_LABELS,
)
from train.dataset import EmotionDataset
from train.metrics import compute_classification_metrics
from train.model import build_model

BATCH_SIZE = 64
EPOCHS = 30
BACKBONE_LR = 1e-4
CLASSIFIER_LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
PATIENCE = 7
LABEL_SMOOTHING = 0.08


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def build_train_transform():
    return T.Compose(
        [
            T.Resize((SAVED_IMG_SIZE, SAVED_IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=0.12,
                        contrast=0.12,
                        saturation=0.08,
                        hue=0.015,
                    )
                ],
                p=0.35,
            ),
            T.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
            T.RandomCrop(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )


def build_eval_transform():
    return T.Compose(
        [
            T.Resize((SAVED_IMG_SIZE, SAVED_IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )


def build_dataloaders(device):
    train_ds = EmotionDataset(split='train', transform=build_train_transform())
    val_ds = EmotionDataset(split='val', transform=build_eval_transform())

    common_loader_args = {
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'pin_memory': device.type == 'cuda',
        'persistent_workers': NUM_WORKERS > 0,
    }

    train_loader = DataLoader(train_ds, shuffle=True, **common_loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader_args)
    return train_ds, val_ds, train_loader, val_loader


def build_loss(train_targets, device):
    counts = np.bincount(np.array(train_targets), minlength=len(TARGET_LABELS))
    imbalance_ratio = counts.max() / max(counts.min(), 1)

    if imbalance_ratio >= 1.25:
        weights = counts.sum() / np.maximum(counts, 1)
        weights = weights / weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        class_weights = None

    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)


def get_optimizer(model):
    classifier_params = list(model.classifier.parameters())
    classifier_param_ids = {id(param) for param in classifier_params}

    backbone_decay = []
    backbone_no_decay = []
    classifier_decay = []
    classifier_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_classifier = id(param) in classifier_param_ids
        is_no_decay = param.ndim == 1 or name.endswith('.bias')

        if is_classifier and is_no_decay:
            classifier_no_decay.append(param)
        elif is_classifier:
            classifier_decay.append(param)
        elif is_no_decay:
            backbone_no_decay.append(param)
        else:
            backbone_decay.append(param)

    return torch.optim.AdamW(
        [
            {'params': backbone_decay, 'lr': BACKBONE_LR, 'weight_decay': WEIGHT_DECAY},
            {'params': backbone_no_decay, 'lr': BACKBONE_LR, 'weight_decay': 0.0},
            {'params': classifier_decay, 'lr': CLASSIFIER_LR, 'weight_decay': WEIGHT_DECAY},
            {'params': classifier_no_decay, 'lr': CLASSIFIER_LR, 'weight_decay': 0.0},
        ]
    )


def run_epoch(model, loader, criterion, optimizer, device, scaler=None, epoch=None, phase='train'):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_samples = 0
    use_amp = device.type == 'cuda'

    y_true = []
    y_pred = []
    y_prob = []

    progress = tqdm(loader, desc=f'Ep {epoch:02d} [{phase}]', leave=False)

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

        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total_samples += labels.size(0)

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())
        y_prob.extend(probabilities.detach().cpu().tolist())

        running_metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        running_loss = total_loss / total_samples
        progress.set_postfix(
            loss=f"{running_loss:.4f}",
            acc=f"{running_metrics['accuracy']:.4f}",
            f1=f"{running_metrics['macro_f1']:.4f}",
        )

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    metrics['loss'] = total_loss / total_samples
    metrics['report'] = classification_report(
        y_true,
        y_pred,
        target_names=TARGET_LABELS,
        output_dict=True,
        zero_division=0,
    )
    return metrics


def save_history(history):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    history_json = LOG_DIR / 'train_history.json'
    with history_json.open('w', encoding='utf-8') as handle:
        json.dump(history, handle, indent=2, ensure_ascii=False)

    history_csv = LOG_DIR / 'train_log.csv'
    with history_csv.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'epoch',
                'train_loss',
                'train_acc',
                'train_macro_f1',
                'train_balanced_acc',
                'val_loss',
                'val_acc',
                'val_macro_f1',
                'val_balanced_acc',
                'val_top2_acc',
                'val_macro_precision',
                'val_macro_recall',
                'lr_backbone',
                'lr_classifier',
                'seconds',
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def save_best_checkpoint(model, metrics, epoch):
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'model_name': MODEL_NAME,
            'model_display_name': MODEL_DISPLAY_NAME,
            'best_epoch': epoch,
            'best_metric_name': 'val_macro_f1',
            'best_metric_value': metrics['macro_f1'],
            'metrics': metrics,
            'img_size': IMG_SIZE,
            'saved_img_size': SAVED_IMG_SIZE,
            'labels': TARGET_LABELS,
        },
        EXPORT_DIR / 'best_model.pth',
    )


def main():
    seed_everything(SEED)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds, _, train_loader, val_loader = build_dataloaders(device)

    model = build_model(pretrained=True).to(device)
    criterion = build_loss(train_ds.targets, device)
    optimizer = get_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    best_val_macro_f1 = 0.0
    epochs_without_improvement = 0
    history = []

    print(
        f'Antrenare {MODEL_DISPLAY_NAME} | device={device} | batch_size={BATCH_SIZE} | epochs={EPOCHS}'
    )
    print(
        f'Input model={IMG_SIZE}x{IMG_SIZE} | imagini salvate={SAVED_IMG_SIZE}x{SAVED_IMG_SIZE} '
        f'| selectie checkpoint=val_macro_f1'
    )

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            epoch=epoch,
            phase='train',
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                scaler=None,
                epoch=epoch,
                phase='val',
            )

        scheduler.step()
        elapsed = time.time() - start_time
        lr_backbone = optimizer.param_groups[0]['lr']
        lr_classifier = optimizer.param_groups[2]['lr']

        row = {
            'epoch': epoch,
            'train_loss': round(train_metrics['loss'], 6),
            'train_acc': round(train_metrics['accuracy'], 6),
            'train_macro_f1': round(train_metrics['macro_f1'], 6),
            'train_balanced_acc': round(train_metrics['balanced_accuracy'], 6),
            'val_loss': round(val_metrics['loss'], 6),
            'val_acc': round(val_metrics['accuracy'], 6),
            'val_macro_f1': round(val_metrics['macro_f1'], 6),
            'val_balanced_acc': round(val_metrics['balanced_accuracy'], 6),
            'val_top2_acc': round(val_metrics['top2_accuracy'], 6),
            'val_macro_precision': round(val_metrics['macro_precision'], 6),
            'val_macro_recall': round(val_metrics['macro_recall'], 6),
            'lr_backbone': round(lr_backbone, 8),
            'lr_classifier': round(lr_classifier, 8),
            'seconds': round(elapsed, 2),
        }
        history.append(row)

        print(
            f"Epoca {epoch:02d} | train_acc={train_metrics['accuracy']:.4f} | "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | val_bal_acc={val_metrics['balanced_accuracy']:.4f} | "
            f"val_top2={val_metrics['top2_accuracy']:.4f} | train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f}"
        )

        if val_metrics['macro_f1'] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics['macro_f1']
            epochs_without_improvement = 0
            save_best_checkpoint(model, val_metrics, epoch)
            print(f'Checkpoint nou salvat cu val_macro_f1={best_val_macro_f1:.4f}')
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print('Early stopping activat.')
            break

    save_history(history)
    print(f'Antrenarea s-a incheiat. Cel mai bun val_macro_f1={best_val_macro_f1:.4f}')


if __name__ == '__main__':
    main()
