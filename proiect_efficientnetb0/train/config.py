from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = PROJECT_DIR.parent / 'archive'
COMBINED_LABELS_CSV = ARCHIVE_DIR / 'combined_labels.csv'

PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed_4classes'
IMAGES_DIR = PROCESSED_DIR / 'images'
META_PATH = PROCESSED_DIR / 'meta.json'

EXPORT_DIR = PROJECT_DIR / 'export'
LOG_DIR = PROJECT_DIR / 'logs'

SEED = 42
MODEL_NAME = 'efficientnet_b0'
MODEL_DISPLAY_NAME = 'EfficientNet-B0'
IMG_SIZE = 224
SAVED_IMG_SIZE = 256
NUM_CLASSES = 4

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

TARGET_LABELS = ['happy', 'neutral', 'sad', 'surprise']
CLASS_TO_IDX = {label: idx for idx, label in enumerate(TARGET_LABELS)}
IDX2LABEL = {idx: label for label, idx in CLASS_TO_IDX.items()}

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
