import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import inspect
import os
import random
import re
from glob import glob

import numpy as np
from PIL import Image
from hailo_sdk_client import ClientRunner

from train.config import EXPORT_DIR, IMAGE_MEAN, IMAGE_STD, IMG_SIZE, SAVED_IMG_SIZE


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compile ONNX -> HEF for Hailo (calibration + normalization + avgpool fix).'
    )
    parser.add_argument(
        '--onnx',
        default=str(EXPORT_DIR / 'emotion_model.onnx'),
        help='Path to ONNX model',
    )
    parser.add_argument(
        '--calib-dir',
        default='data/processed_4classes/images/train',
        help='Folder with calibration images',
    )
    parser.add_argument('--model-name', default='emotion_model', help='Internal model name')
    parser.add_argument(
        '--hef-out',
        default=str(EXPORT_DIR / 'emotion_model.hef'),
        help='Output HEF path',
    )
    parser.add_argument('--hw-arch', default='hailo8', help='Target arch: hailo8 / hailo8l etc.')
    parser.add_argument('--input-w', type=int, default=IMG_SIZE, help='Input width')
    parser.add_argument('--input-h', type=int, default=IMG_SIZE, help='Input height')
    parser.add_argument(
        '--resize-size',
        type=int,
        default=SAVED_IMG_SIZE,
        help='Resize size before center crop during calibration preprocessing.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=1024,
        help='Max calibration images to load (0 = load all). Recommended >= 1024.',
    )
    parser.add_argument('--seed', type=int, default=123, help='Shuffle seed for picking calib images')
    parser.add_argument(
        '--gpu',
        default='0',
        help="CUDA_VISIBLE_DEVICES (e.g. 0). Use '' to force CPU.",
    )
    parser.add_argument(
        '--avgpool-layer',
        default='auto',
        help="AvgPool layer name for global_avgpool_reduction. Use 'auto' to detect it or 'none' to disable it.",
    )
    parser.add_argument(
        '--division',
        type=int,
        nargs=2,
        default=[2, 2],
        help='global_avgpool_reduction division factors, e.g. 2 2 or 4 4 if still fails.',
    )
    parser.add_argument(
        '--wsl-libcuda',
        action='store_true',
        help='Prepends /usr/lib/wsl/lib to LD_LIBRARY_PATH (useful in WSL2).',
    )
    return parser.parse_args()


def list_images(calib_dir: str):
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
    files = []
    for ext in exts:
        files.extend(glob(str(Path(calib_dir) / '*' / ext)))
        files.extend(glob(str(Path(calib_dir) / ext)))
    return sorted(files)


def preprocess_calibration_image(image: Image.Image, resize_size: int, input_w: int, input_h: int):
    image = image.convert('RGB')
    image = image.resize((resize_size, resize_size), Image.Resampling.BILINEAR)

    left = max((resize_size - input_w) // 2, 0)
    top = max((resize_size - input_h) // 2, 0)
    right = left + input_w
    bottom = top + input_h
    image = image.crop((left, top, right, bottom))
    return np.asarray(image, dtype=np.uint8)


def load_calib_dataset(
    calib_dir: str,
    resize_size: int,
    input_w: int,
    input_h: int,
    limit: int,
    seed: int,
) -> np.ndarray:
    paths = list_images(calib_dir)
    if not paths:
        raise FileNotFoundError(f'Nu am gasit imagini in: {calib_dir}')

    random.seed(seed)
    random.shuffle(paths)

    if limit and limit > 0 and len(paths) > limit:
        paths = paths[:limit]

    print(f'--- Incarcare imagini de calibrare din: {calib_dir} ---')
    print(f'Am gasit {len(paths)} imagini (dupa limit). Procesez...')

    imgs = []
    bad = 0
    effective_resize = max(resize_size, input_w, input_h)

    for index, file_path in enumerate(paths):
        try:
            with Image.open(file_path) as image:
                imgs.append(
                    preprocess_calibration_image(
                        image=image,
                        resize_size=effective_resize,
                        input_w=input_w,
                        input_h=input_h,
                    )
                )
        except Exception:
            bad += 1
            continue

        if (index + 1) % 100 == 0:
            print(f'  ... {index + 1}/{len(paths)} (citite ok: {len(imgs)})')

    if not imgs:
        raise RuntimeError('Nu am reusit sa citesc nicio imagine valida pentru calibrare.')

    if bad:
        print(f'[warning] {bad} imagini nu s-au putut citi si au fost ignorate.')

    data = np.stack(imgs, axis=0).astype(np.uint8, copy=False)
    print(f'Dataset calibrare pregatit. Shape: {data.shape} | Tip: {data.dtype}')
    return data


def detect_avgpool_layer(onnx_path: Path):
    text = onnx_path.read_bytes().decode('latin1', errors='ignore')
    matches = re.findall(r'([A-Za-z0-9_./-]*avgpool/GlobalAveragePool)', text)
    if not matches:
        return None

    if '/avgpool/GlobalAveragePool' in matches:
        return '/avgpool/GlobalAveragePool'
    return matches[-1]


def build_model_script(onnx_path: Path, avgpool_layer: str, division_factors):
    div0, div1 = int(division_factors[0]), int(division_factors[1])
    norm_mean = [round(255.0 * value, 6) for value in IMAGE_MEAN]
    norm_std = [round(255.0 * value, 6) for value in IMAGE_STD]

    selected_avgpool_layer = avgpool_layer
    if selected_avgpool_layer == 'auto':
        selected_avgpool_layer = detect_avgpool_layer(onnx_path)
    elif selected_avgpool_layer == 'none':
        selected_avgpool_layer = None

    script_lines = [f'normalization1 = normalization({norm_mean}, {norm_std})']
    if selected_avgpool_layer:
        script_lines.append(
            'pre_quantization_optimization('
            f'global_avgpool_reduction, layers={selected_avgpool_layer}, division_factors=[{div0}, {div1}]'
            ')'
        )
    else:
        print('[warning] Nu am detectat un strat avgpool final. Continui fara global_avgpool_reduction.')

    print(f'[info] Normalizare Hailo aplicata cu mean={norm_mean} si std={norm_std}')
    if selected_avgpool_layer:
        print(f'[info] AvgPool layer folosit pentru optimizare: {selected_avgpool_layer}')

    return '\n'.join(script_lines) + '\n'


def optimize_with_all_calib_entries(runner: ClientRunner, calib_data: np.ndarray):
    num_entries = int(len(calib_data))
    print(f'[info] CALIB LEN REAL: {num_entries}')

    signature = inspect.signature(runner.optimize)
    params = set(signature.parameters.keys())
    candidates = [
        'calib_dataset_size',
        'max_calib_entries',
        'dataset_size',
        'calib_data_count',
        'calib_entries',
        'num_calib_entries',
        'num_entries',
    ]

    kwargs = {}
    for candidate in candidates:
        if candidate in params:
            kwargs[candidate] = num_entries
            print(f'[info] Fortez optimize(..., {candidate}={num_entries}) ca sa nu foloseasca doar 64.')
            break

    runner.optimize(calib_data, **kwargs)


def main():
    args = parse_args()
    onnx_path = Path(args.onnx)
    hef_path = Path(args.hef_out)

    if not onnx_path.exists():
        raise FileNotFoundError(
            f'Nu exista modelul ONNX {onnx_path}. Ruleaza mai intai python -m export.export_onnx.'
        )

    if args.wsl_libcuda:
        os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print(f'[info] python: {sys.executable}')
    try:
        import tensorflow as tf

        print(f"[info] TF={tf.__version__} | GPUs={tf.config.list_physical_devices('GPU')}")
    except Exception as exc:
        print(f'[warning] Nu pot importa TensorFlow pentru diagnostic: {exc}')

    print(f"[info] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    print(f"[info] LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')!r}")

    print(f'1. Initializez ClientRunner pentru {args.hw_arch}...')
    runner = ClientRunner(hw_arch=args.hw_arch)

    print(f'2. Convertesc ONNX ({onnx_path}) -> HAR intern...')
    runner.translate_onnx_model(str(onnx_path), args.model_name)

    print('2.5. Aplic model script (normalizare + avgpool fix)...')
    model_script = build_model_script(onnx_path, args.avgpool_layer, args.division)
    runner.load_model_script(model_script)

    calib_data = load_calib_dataset(
        calib_dir=args.calib_dir,
        resize_size=args.resize_size,
        input_w=args.input_w,
        input_h=args.input_h,
        limit=args.limit,
        seed=args.seed,
    )

    print('3. Optimizez modelul (Quantization)...')
    optimize_with_all_calib_entries(runner, calib_data)

    print('4. Compilez fisierul HEF...')
    hef_bytes = runner.compile()

    hef_path.parent.mkdir(parents=True, exist_ok=True)
    hef_path.write_bytes(hef_bytes)
    print(f'\nSUCCES! Fisierul a fost salvat: {hef_path.resolve()}')


if __name__ == '__main__':
    main()
