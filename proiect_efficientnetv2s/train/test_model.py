import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms import InterpolationMode

from train.config import EXPORT_DIR, IDX2LABEL, IMG_SIZE
from train.model import build_model

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description="Testeaza o imagine individuala.")
    parser.add_argument("--image", required=True, help="Calea catre imaginea testata.")
    parser.add_argument("--top-k", type=int, default=4, help="Cate predictii sa fie afisate.")
    return parser.parse_args()


def build_transform():
    return T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )


def load_model(device):
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


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    transform = build_transform()

    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    probabilities = torch.softmax(model(tensor), dim=1)[0]
    top_k = min(args.top_k, len(IDX2LABEL))
    scores, indices = torch.topk(probabilities, k=top_k)

    print(f"Predictii pentru {args.image}:")
    for score, index in zip(scores.tolist(), indices.tolist()):
        print(f"- {IDX2LABEL[index]}: {score * 100:.2f}%")


if __name__ == "__main__":
    main()

