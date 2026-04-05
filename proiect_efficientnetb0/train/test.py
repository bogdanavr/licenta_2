import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from train.config import (
    EXPORT_DIR,
    IMAGE_MEAN,
    IMAGE_STD,
    IMG_SIZE,
    LOG_DIR,
    MODEL_DISPLAY_NAME,
    NUM_CLASSES,
    SAVED_IMG_SIZE,
    TARGET_LABELS,
)
from train.dataset import EmotionDataset
from train.metrics import compute_classification_metrics
from train.model import build_model


def build_eval_transform():
    return T.Compose(
        [
            T.Resize((SAVED_IMG_SIZE, SAVED_IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )


def load_checkpoint(device):
    checkpoint_path = EXPORT_DIR / 'best_model.pth'
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f'Nu exista checkpoint-ul {checkpoint_path}. Ruleaza mai intai train/train.py.'
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def save_confusion_matrix(matrix, labels):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = LOG_DIR / 'confusion_matrix.png'

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, cmap='Blues')
    plt.colorbar(image, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predictie')
    ax.set_ylabel('Eticheta reala')
    ax.set_title('Matrice de confuzie normalizata')

    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = 'white' if matrix[i, j] > threshold else 'black'
            ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color=color)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, checkpoint = load_checkpoint(device)

    test_ds = EmotionDataset(split='test', transform=build_eval_transform())
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == 'cuda',
        persistent_workers=True,
    )

    y_true = []
    y_pred = []
    y_prob = []

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predictions.tolist())
        y_prob.extend(probabilities.tolist())

    label_names = [TARGET_LABELS[idx] for idx in range(NUM_CLASSES)]
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = LOG_DIR / 'test_metrics.json'
    report_path = LOG_DIR / 'classification_report.json'

    payload = {
        'model_display_name': checkpoint.get('model_display_name', MODEL_DISPLAY_NAME),
        'best_epoch': checkpoint.get('best_epoch'),
        'selection_metric': checkpoint.get('best_metric_name'),
        'selection_metric_value': checkpoint.get('best_metric_value'),
        'test_metrics': metrics,
    }

    with metrics_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    with report_path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    matrix = confusion_matrix(y_true, y_pred, normalize='true')
    save_confusion_matrix(matrix, label_names)

    print('Metrici test:')
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f'Raportul detaliat a fost salvat in {report_path}')
    print(f'Metricile test au fost salvate in {metrics_path}')
    print(f"Matricea de confuzie a fost salvata in {LOG_DIR / 'confusion_matrix.png'}")


if __name__ == '__main__':
    main()
