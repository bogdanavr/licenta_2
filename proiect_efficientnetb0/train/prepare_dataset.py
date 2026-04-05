"""Pregateste subsetul cu 4 emotii folosind combined_labels.csv ca sursa autoritara."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict

import cv2
from tqdm import tqdm

from train.config import (
    ARCHIVE_DIR,
    COMBINED_LABELS_CSV,
    IMG_SIZE,
    IMAGES_DIR,
    META_PATH,
    MODEL_DISPLAY_NAME,
    PROCESSED_DIR,
    SAVED_IMG_SIZE,
    SEED,
    SUPPORTED_EXTENSIONS,
    TARGET_LABELS,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from train.preprocessing import preprocess_face_bgr


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pregateste subsetul cu 4 clase pentru proiectul EfficientNet-B0.'
    )
    parser.add_argument(
        '--sources',
        nargs='+',
        choices=['original', 'augmented'],
        default=['original', 'augmented'],
        help='Sursele incluse in subset.',
    )
    return parser.parse_args()


def reset_processed_dir():
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def load_records(allowed_sources):
    if not COMBINED_LABELS_CSV.exists():
        raise FileNotFoundError(f'Nu exista fisierul {COMBINED_LABELS_CSV}')

    records = []
    # CSV-ul ramane sursa de adevar pentru etichete si cai, deoarece folderele brute contin inconsistente.
    with COMBINED_LABELS_CSV.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = row['label'].strip()
            source = row['source'].strip()
            rel_path = row['filepath'].strip()
            src_path = ARCHIVE_DIR / rel_path

            if label not in TARGET_LABELS or source not in allowed_sources:
                continue
            if src_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if not src_path.exists():
                continue

            records.append(
                {
                    'label': label,
                    'source': source,
                    'rel_path': rel_path.replace('\\', '/'),
                    'src_path': src_path,
                }
            )

    if not records:
        raise RuntimeError('Nu am gasit inregistrari valide pentru cele 4 clase.')

    return records


def stratified_split(records):
    rng = random.Random(SEED)
    grouped = defaultdict(list)
    for record in records:
        grouped[record['label']].append(record)

    split_records = {'train': [], 'val': [], 'test': []}
    for label in TARGET_LABELS:
        items = grouped[label]
        rng.shuffle(items)

        total = len(items)
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        split_records['train'].extend(items[:train_end])
        split_records['val'].extend(items[train_end:val_end])
        split_records['test'].extend(items[val_end:])

    for split_name in split_records:
        rng.shuffle(split_records[split_name])

    return split_records


def save_processed_image(src_path, dst_path):
    image = cv2.imread(str(src_path))
    if image is None:
        raise RuntimeError(f'Nu am putut citi imaginea {src_path}')

    processed, found_face = preprocess_face_bgr(image, SAVED_IMG_SIZE)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(dst_path), processed)
    if not success:
        raise RuntimeError(f'Nu am putut salva imaginea procesata la {dst_path}')

    return found_face


def materialize_split(split_name, items):
    split_dir = IMAGES_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        'total': 0,
        'face_detected': 0,
        'fallback_center_crop': 0,
        'labels': Counter(),
        'sources': Counter(),
    }

    for index, item in enumerate(tqdm(items, desc=f'Pregatire {split_name}', unit='img'), start=1):
        extension = item['src_path'].suffix.lower()
        filename = f'{index:05d}{extension}'
        output_path = split_dir / item['label'] / filename

        found_face = save_processed_image(item['src_path'], output_path)

        stats['total'] += 1
        stats['labels'][item['label']] += 1
        stats['sources'][item['source']] += 1
        if found_face:
            stats['face_detected'] += 1
        else:
            stats['fallback_center_crop'] += 1

    return stats


def build_meta(split_stats, sources):
    meta = {
        'dataset_csv': str(COMBINED_LABELS_CSV),
        'archive_dir': str(ARCHIVE_DIR),
        'model': MODEL_DISPLAY_NAME,
        'target_labels': TARGET_LABELS,
        'sources': list(sources),
        'ratios': {
            'train': TRAIN_RATIO,
            'val': VAL_RATIO,
            'test': TEST_RATIO,
        },
        'preprocessing': {
            'face_crop': True,
            'face_detector': 'opencv_haar_frontalface_default',
            'fallback_when_no_face': 'center_square_crop',
            'square_padding': 'border_replicate',
            'saved_image_size': [SAVED_IMG_SIZE, SAVED_IMG_SIZE],
            'model_input_size': [IMG_SIZE, IMG_SIZE],
        },
        'splits': {},
    }

    for split_name, stats in split_stats.items():
        meta['splits'][split_name] = {
            'total': int(stats['total']),
            'face_detected': int(stats['face_detected']),
            'fallback_center_crop': int(stats['fallback_center_crop']),
            'labels': {label: int(stats['labels'].get(label, 0)) for label in TARGET_LABELS},
            'sources': dict(stats['sources']),
        }

    with META_PATH.open('w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    reset_processed_dir()

    records = load_records(set(args.sources))
    split_records = stratified_split(records)

    split_stats = {}
    for split_name, items in split_records.items():
        split_stats[split_name] = materialize_split(split_name, items)

    build_meta(split_stats, args.sources)

    print(f'Subset pregatit in: {PROCESSED_DIR}')
    print(
        'Train/Val/Test: '
        f"{split_stats['train']['total']}/{split_stats['val']['total']}/{split_stats['test']['total']}"
    )
    for split_name, stats in split_stats.items():
        print(
            f"{split_name}: face_detected={stats['face_detected']} | "
            f"fallback_center_crop={stats['fallback_center_crop']}"
        )


if __name__ == '__main__':
    main()
