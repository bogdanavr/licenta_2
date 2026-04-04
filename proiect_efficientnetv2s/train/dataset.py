import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from torch.utils.data import Dataset

from train.config import CLASS_TO_IDX, IMAGES_DIR, TARGET_LABELS


class EmotionDataset(Dataset):
    def __init__(self, split: str = "train", transform=None):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Split invalid: {split}")

        self.transform = transform
        self.samples = []
        self.targets = []

        split_dir = IMAGES_DIR / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Lipseste directorul {split_dir}. Ruleaza mai intai python -m train.prepare_dataset."
            )

        for label in TARGET_LABELS:
            class_dir = split_dir / label
            if not class_dir.exists():
                continue

            for image_path in sorted(path for path in class_dir.iterdir() if path.is_file()):
                target = CLASS_TO_IDX[label]
                self.samples.append((image_path, target))
                self.targets.append(target)

        if not self.samples:
            raise RuntimeError(f"Nu am gasit imagini in {split_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
