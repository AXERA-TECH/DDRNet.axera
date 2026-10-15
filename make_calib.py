# Copyright (c) OpenMMLab. All rights reserved.
"""Build the quantization calibration set for the exported DDRNet ONNX models.

Every image found under ``--img-dir`` (recursively) is preprocessed with the
exact ``preprocess`` used by ``ddrnet_onnx_eval.py`` (BGR read, resize to
``--input-size`` H W with INTER_LINEAR, BGR->RGB, mean/std normalization,
float32, NCHW with batch dim) and the raw tensor is written to
``<out-dir>/<stem>.bin``. With ``--resize-only`` the channel swap and
normalization are skipped (resize-only, values 0-255). With
``--save-images`` the resized images are saved as PNG instead of tensors.
All outputs are then archived into a flat tar (``./<stem>.<ext>`` entries)
for the pulsar2 quantization tool.

Examples:
    python ddrnet_make_calib.py
    python ddrnet_make_calib.py --input-size 1024 2048
    python ddrnet_make_calib.py --input-size 1024 2048 --resize-only
    python ddrnet_make_calib.py --input-size 1024 2048 --save-images
"""
import argparse
import glob
import os
import tarfile

import cv2
import numpy as np


def preprocess(img_bgr, size):
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(
        img_bgr, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    rgb = resized[:, :, ::-1]
    # rgb = resized.astype(np.float32)[:, :, ::-1]
    # normalized = (rgb - MEAN) / STD
    # inp = normalized.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(rgb)[None]


def resize_only(img_bgr, size):
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(
        img_bgr, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    inp = resized.astype(np.float32).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(inp)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build DDRNet calibration bins from Cityscapes images')
    parser.add_argument(
        '--img-dir',
        default='../dataset/cityscapes/val',
        help='image dir searched recursively for *.png')
    parser.add_argument(
        '--out-dir',
        default=None,
        help='directory the .bin files are written to; defaults to '
        '../dataset/cityscapes/ddrnet_calib_<H>x<W>-bin')
    parser.add_argument(
        '--input-size',
        type=int,
        nargs=2,
        default=[1024, 1024],
        help='model input size H W')
    parser.add_argument(
        '--resize-only',
        action='store_true',
        help='only resize to the input size (BGR, float32, 0-255); skip the '
        'BGR->RGB and mean/std normalization of ddrnet_onnx_eval.preprocess')
    parser.add_argument(
        '--save-images',
        action='store_true',
        help='save the resized images as PNG instead of float32 .bin tensors')
    parser.add_argument(
        '--tar', default=None, help='tar path; defaults to <out-dir>.tar')
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='only use the first N images (for a quick check)')
    return parser.parse_args()


def main():
    args = parse_args()
    size = tuple(args.input_size)
    prep = resize_only if args.resize_only else preprocess
    tag = '-resize' if args.resize_only else ''
    ext = 'bin'
    suffix = '-bin'
    if args.save_images:
        tag = '-resize'
        ext = 'png'
        suffix = '-img'
    out_dir = args.out_dir or (
        f'../dataset/cityscapes/ddrnet_calib_{size[0]}x{size[1]}{tag}{suffix}')
    tar_path = args.tar or out_dir.rstrip('/') + '.tar'

    img_paths = sorted(
        glob.glob(os.path.join(args.img_dir, '**', '*.png'), recursive=True))
    assert img_paths, f'no images found under {args.img_dir}'
    if args.limit:
        img_paths = img_paths[:args.limit]
    print(f'[data] img_dir={args.img_dir} images={len(img_paths)} '
          f'input_size={size}')
    print(f'[data] out_dir={out_dir} tar={tar_path}')

    os.makedirs(out_dir, exist_ok=True)
    for i, img_path in enumerate(img_paths, 1):
        img = cv2.imread(img_path)
        assert img is not None, f'failed to read {img_path}'
        stem = os.path.basename(img_path)[:-4]
        if stem.endswith('_leftImg8bit'):
            stem = stem[:-len('_leftImg8bit')]
        if args.save_images:
            resized = cv2.resize(
                img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(out_dir, f'{stem}.{ext}'), resized)
        else:
            tensor = prep(img, size)
            tensor.astype(np.float32).tofile(
                os.path.join(out_dir, f'{stem}.{ext}'))
        if i % 50 == 0 or i == len(img_paths):
            print(f'[calib] {i}/{len(img_paths)}')

    stem = os.path.basename(img_paths[0])[:-4]
    if stem.endswith('_leftImg8bit'):
        stem = stem[:-len('_leftImg8bit')]
    if args.save_images:
        saved = cv2.imread(os.path.join(out_dir, f'{stem}.{ext}'))
        expect = cv2.resize(
            cv2.imread(img_paths[0]), (size[1], size[0]),
            interpolation=cv2.INTER_LINEAR)
        assert np.array_equal(saved, expect), 'saved image mismatch'
        print(f'[check] {stem}.{ext} shape={saved.shape[:2]} '
              f'range=[{saved.min()}, {saved.max()}]')
    else:
        data = np.fromfile(
            os.path.join(out_dir, f'{stem}.{ext}'),
            dtype=np.float32).reshape(1, 3, *size)
        expect = prep(cv2.imread(img_paths[0]), size)
        np.testing.assert_allclose(data, expect, rtol=1e-5, atol=1e-5)
        print(f'[check] {stem}.{ext} shape={data.shape} '
              f'range=[{data.min():.3f}, {data.max():.3f}]')

    names = sorted(f for f in os.listdir(out_dir) if f.endswith(f'.{ext}'))
    with tarfile.open(tar_path, 'w') as tar:
        for name in names:
            tar.add(os.path.join(out_dir, name), arcname='./' + name)
    print(f'[tar] archived {len(names)} {ext} files to {tar_path} '
          f'({os.path.getsize(tar_path) / 2**30:.2f} GiB)')


if __name__ == '__main__':
    main()
