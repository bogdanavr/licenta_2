import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

import torch

from train.config import (
    EXPORT_DIR,
    IMAGE_MEAN,
    IMAGE_STD,
    IMG_SIZE,
    MODEL_DISPLAY_NAME,
    MODEL_NAME,
    TARGET_LABELS,
)
from train.model import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Converteste checkpoint-ul .pth in model .onnx pentru Raspberry Pi / Hailo.'
    )
    parser.add_argument(
        '--checkpoint',
        default=str(EXPORT_DIR / 'best_model.pth'),
        help='Calea catre checkpoint-ul PyTorch (.pth).',
    )
    parser.add_argument(
        '--output',
        default=str(EXPORT_DIR / 'emotion_model.onnx'),
        help='Calea unde va fi salvat modelul ONNX.',
    )
    parser.add_argument(
        '--metadata-out',
        default=str(EXPORT_DIR / 'emotion_model.metadata.json'),
        help='Fisier JSON cu metadatele exportului.',
    )
    parser.add_argument('--img-size', type=int, default=IMG_SIZE, help='Dimensiunea inputului modelului.')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch-size fix pentru export.')
    parser.add_argument('--opset', type=int, default=13, help='ONNX opset folosit la export.')
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device folosit la export. Pentru compatibilitate maxima, lasa cpu.',
    )
    parser.add_argument(
        '--dynamic-batch',
        action='store_true',
        help='Activeaza batch dinamic in ONNX. Pentru Hailo, batch fix 1 este de obicei mai sigur.',
    )
    parser.add_argument(
        '--skip-check',
        action='store_true',
        help='Nu mai ruleaza validarea ONNX cu onnx.checker dupa export.',
    )
    return parser.parse_args()


def load_checkpoint_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f'Nu exista checkpoint-ul {checkpoint_path}. Ruleaza mai intai train/train.py.'
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    model = build_model(pretrained=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def export_model(
    model,
    output_path: Path,
    img_size: int,
    batch_size: int,
    opset: int,
    device: torch.device,
    dynamic_batch: bool,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch'},
            'logits': {0: 'batch'},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['logits'],
            dynamic_axes=dynamic_axes,
        )


def maybe_validate_onnx(output_path: Path):
    try:
        import onnx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Lipseste pachetul 'onnx'. Instaleaza-l in mediul de export ca sa poti genera si verifica modelul."
        ) from exc

    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)


def save_metadata(metadata_path: Path, checkpoint_path: Path, output_path: Path, checkpoint, args):
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'checkpoint_path': str(checkpoint_path),
        'onnx_path': str(output_path),
        'model_name': checkpoint.get('model_name', MODEL_NAME),
        'model_display_name': checkpoint.get('model_display_name', MODEL_DISPLAY_NAME),
        'best_epoch': checkpoint.get('best_epoch'),
        'labels': checkpoint.get('labels', TARGET_LABELS),
        'img_size': checkpoint.get('img_size', args.img_size),
        'image_mean': IMAGE_MEAN,
        'image_std': IMAGE_STD,
        'opset': args.opset,
        'dynamic_batch': args.dynamic_batch,
    }
    with metadata_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    metadata_path = Path(args.metadata_out)

    model, checkpoint = load_checkpoint_model(checkpoint_path, device)
    export_model(
        model=model,
        output_path=output_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        opset=args.opset,
        device=device,
        dynamic_batch=args.dynamic_batch,
    )

    if not args.skip_check:
        maybe_validate_onnx(output_path)

    save_metadata(metadata_path, checkpoint_path, output_path, checkpoint, args)

    print(f'Checkpoint incarcat: {checkpoint_path.resolve()}')
    print(f'Model ONNX salvat la: {output_path.resolve()}')
    print(f'Metadate salvate la: {metadata_path.resolve()}')
    print('Preprocesare asteptata: RGB -> float32 -> normalize(ImageNet mean/std) -> NCHW')


if __name__ == '__main__':
    main()
