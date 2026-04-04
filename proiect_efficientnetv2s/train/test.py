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

from train.config import EXPORT_DIR, IDX2LABEL, IMG_SIZE, LOG_DIR, NUM_CLASSES
from train.dataset import EmotionDataset
from train.model import build_model

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_eval_transform():
    return T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )


def load_checkpoint(device):
    checkpoint_path = EXPORT_DIR / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Nu exista checkpoint-ul {checkpoint_path}. Ruleaza mai intai train/train.py."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def save_confusion_matrix(matrix, labels):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = LOG_DIR / "confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(image, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predictie")
    ax.set_ylabel("Eticheta reala")
    ax.set_title("Matrice de confuzie")

    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > threshold else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(device)

    test_ds = EmotionDataset(split="test", transform=build_eval_transform())
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )

    y_true = []
    y_pred = []

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        predictions = logits.argmax(dim=1).cpu().numpy()

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predictions.tolist())

    label_names = [IDX2LABEL[idx] for idx in range(NUM_CLASSES)]
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    report_path = LOG_DIR / "classification_report.json"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    matrix = confusion_matrix(y_true, y_pred, normalize="true")
    save_confusion_matrix(matrix, label_names)

    print("Raport evaluare:")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Raportul a fost salvat in {report_path}")
    print(f"Matricea de confuzie a fost salvata in {LOG_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()

