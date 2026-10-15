# Copyright (c) OpenMMLab. All rights reserved.
"""Standalone Cityscapes inference/evaluation for exported DDRNet ONNX models.

Runs the ONNX model over a directory of images and dumps structured outputs
for downstream prediction-vs-GT comparison. Preprocessing (resize, BGR->RGB,
mean/std normalization) is done in this script; the ONNX graph expects
normalized RGB input:

    <out-dir>/
    ├── meta.json                     # run metadata
    ├── pred/<city>/<stem>_pred.png   # predicted label ids (uint8)
    ├── metrics.json                  # only when GT is available
    └── metrics.csv                   # only when GT is available

Official metrics (optional): 19-class IoU/Acc, mIoU and aAcc with
ignore_index=255, computed against ``*_gtFine_labelTrainIds.png`` GT when
``--gt-dir`` is given (or auto-detected); without GT only predictions are
dumped.

Examples:
    # with GT: predictions + metrics
    python tools/deployment/ddrnet_onnx_eval.py \
        --model runs/ddrnet23-slim_cityscapes-2048x1024.onnx \
        --img-dir /home/heqi/dataset/cityscapes/leftImg8bit/val \
        --gt-dir /home/heqi/dataset/cityscapes/gtFine/val \
        --device cpu --save-vis 5

    # without GT: only structured predictions are dumped
    python tools/deployment/ddrnet_onnx_eval.py \
        --model runs/ddrnet23-slim_cityscapes-2048x1024.onnx \
        --img-dir /path/to/images --gt-dir none
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
    [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

NUM_CLASSES = 19
IGNORE_INDEX = 255


def parse_args():
    parser = argparse.ArgumentParser(
        description='Cityscapes inference/evaluation for DDRNet ONNX models')
    parser.add_argument(
        '--model',
        default='onnx/ddrnet23_cityscapes-2048x1024.onnx',
        help='path to the ONNX model')
    parser.add_argument(
        '--img-dir',
        default='../dataset/cityscapes/val',
        help='image dir; defaults to <data-root>/leftImg8bit/<split>')
    parser.add_argument(
        '--gt-dir',
        default=None,
        help='GT dir with *_gtFine_labelTrainIds.png; optional, when not '
        'given it uses <data-root>/gtFine/<split> if that dir exists, '
        'otherwise metrics are skipped; pass "none" to force skipping')
    parser.add_argument(
        '--data-root',
        default='../dataset/cityscapes',
        help='Cityscapes root used when --img-dir/--gt-dir are not given')
    parser.add_argument('--split', default='val')
    parser.add_argument(
        '--out-dir',
        default='onnx-vis',
        help='structured output dir; defaults to '
        'runs/onnx_pred/<model_stem>; pass "none" to disable dumping')
    parser.add_argument(
        '--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument(
        '--input-size',
        type=int,
        nargs=2,
        default=None,
        help='override input size H W (only for dynamic-shape models)')
    parser.add_argument(
        '--limit',
        type=int,
        default=5,
        help='evaluate only the first N images (for a quick check)')
    parser.add_argument(
        '--save-vis',
        type=int,
        default=5,
        help='save blended visualizations for the first N images')
    parser.add_argument(
        '--vis-dir', default='onnx-vis', help='visualization dir')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='blend weight of the colorized prediction, in [0, 1]')
    parser.add_argument(
        '--ref-miou',
        type=float,
        default=None,
        help='optional reference mIoU to show the difference')
    return parser.parse_args()


def get_model_input_size(session, fallback=(1024, 2048)):
    shape = session.get_inputs()[0].shape
    h, w = shape[2], shape[3]
    if isinstance(h, int) and isinstance(w, int):
        return h, w
    return fallback


def preprocess(img_bgr, size):
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(
        img_bgr, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    rgb = resized.astype(np.float32)[:, :, ::-1]
    normalized = (rgb - MEAN) / STD
    inp = normalized.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(inp)


def postprocess(seg_logits, ori_shape):
    logits = seg_logits[0]
    if logits.shape[1:] != tuple(ori_shape):
        logits = np.stack([
            cv2.resize(
                logits[c], (ori_shape[1], ori_shape[0]),
                interpolation=cv2.INTER_LINEAR)
            for c in range(logits.shape[0])
        ])
    return logits.argmax(0).astype(np.uint8)


def compute_metrics(conf):
    intersection = np.diag(conf).astype(np.float64)
    union = (conf.sum(1) + conf.sum(0) - np.diag(conf)).astype(np.float64)
    gt_pixels = conf.sum(1).astype(np.float64)
    iou = np.divide(
        intersection, union,
        out=np.full(NUM_CLASSES, np.nan),
        where=union > 0) * 100
    acc = np.divide(
        intersection, gt_pixels,
        out=np.full(NUM_CLASSES, np.nan),
        where=gt_pixels > 0) * 100
    miou = float(np.nanmean(iou))
    total = conf.sum()
    aacc = float(intersection.sum() / total * 100) if total > 0 else float('nan')
    return iou, acc, miou, aacc


def dump_metrics(out_dir, conf, iou, acc, miou, aacc):
    metrics = {
        'mIoU': miou,
        'aAcc': aacc,
        'IoU': {
            name: float(iou[cid])
            for cid, name in enumerate(CITYSCAPES_CLASSES)
        },
        'Acc': {
            name: float(acc[cid])
            for cid, name in enumerate(CITYSCAPES_CLASSES)
        },
        'confusion_matrix': conf.tolist()
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, 'metrics.csv'), 'w') as f:
        f.write('class,iou,acc\n')
        for cid, name in enumerate(CITYSCAPES_CLASSES):
            f.write(f'{name},{iou[cid]:.2f},{acc[cid]:.2f}\n')
        f.write(f'mIoU,{miou:.2f},\n')
        f.write(f'aAcc,,{aacc:.2f}\n')


def print_metrics(iou, acc, miou, aacc, num_images, avg_ms, ref_miou):
    print(f'{"class":<15s} {"IoU":>7s} {"Acc":>7s}')
    for cid, name in enumerate(CITYSCAPES_CLASSES):
        print(f'{name:<15s} {iou[cid]:7.2f} {acc[cid]:7.2f}')
    print(f'[result] mIoU={miou:.2f} aAcc={aacc:.2f} '
          f'images={num_images} avg_infer={avg_ms:.1f}ms')
    if ref_miou is not None:
        print(f'[result] reference mIoU={ref_miou:.2f} '
              f'diff={miou - ref_miou:+.2f}')


def main():
    args = parse_args()
    alpha = min(max(args.alpha, 0.0), 1.0)

    providers = ['CPUExecutionProvider']
    if args.device == 'cuda':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(args.model, providers=providers)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    size = tuple(args.input_size) if args.input_size else \
        get_model_input_size(sess)
    print(f'[model] {args.model}')
    print(f'[model] providers: {sess.get_providers()} input_size={size}')

    img_dir = args.img_dir or os.path.join(args.data_root, 'leftImg8bit',
                                           args.split)
    if args.gt_dir is None:
        candidate = os.path.join(args.data_root, 'gtFine', args.split)
        gt_dir = candidate if os.path.isdir(candidate) else None
    else:
        gt_dir = None if args.gt_dir.lower() == 'none' else args.gt_dir
    if args.out_dir is None:
        model_stem = os.path.splitext(os.path.basename(args.model))[0]
        out_dir = os.path.join('runs', 'onnx_pred', model_stem)
    elif args.out_dir.lower() == 'none':
        out_dir = None
    else:
        out_dir = args.out_dir
    if out_dir:
        os.makedirs(os.path.join(out_dir, 'pred'), exist_ok=True)

    img_paths = sorted(
        glob.glob(os.path.join(img_dir, '**', '*.png'), recursive=True))
    assert img_paths, f'no images found under {img_dir}'
    if args.limit:
        img_paths = img_paths[:args.limit]
    print(f'[data] img_dir={img_dir} images={len(img_paths)}')
    print(f'[data] gt_dir={gt_dir} out_dir={out_dir}')

    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    times = []
    saved = 0
    num_gt = 0
    for i, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        assert img is not None, f'failed to read {img_path}'
        ori_shape = img.shape[:2]

        inp = preprocess(img, size)
        t0 = time.perf_counter()
        seg_logits = sess.run([output_name], {input_name: inp})[0]
        times.append(time.perf_counter() - t0)
        pred = postprocess(seg_logits, ori_shape)

        rel = os.path.relpath(img_path, img_dir)
        rel_parent = os.path.dirname(rel)
        stem = os.path.basename(img_path)[:-4]
        if stem.endswith('_leftImg8bit'):
            stem = stem[:-len('_leftImg8bit')]

        if out_dir:
            pred_path = os.path.join(out_dir, 'pred', rel_parent,
                                     f'{stem}_pred.png')
            os.makedirs(os.path.dirname(pred_path), exist_ok=True)
            cv2.imwrite(pred_path, pred)

        if saved < args.save_vis:
            color = CITYSCAPES_PALETTE[pred]
            vis = (img.astype(np.float32) * (1 - alpha) +
                   color.astype(np.float32) * alpha).astype(np.uint8)
            vis_path = os.path.join(args.vis_dir, rel_parent,
                                    f'{stem}_vis.png')
            os.makedirs(os.path.dirname(vis_path), exist_ok=True)
            cv2.imwrite(vis_path, vis)
            saved += 1

        if gt_dir:
            gt_path = os.path.join(gt_dir, rel_parent,
                                   f'{stem}_gtFine_labelTrainIds.png')
            if os.path.isfile(gt_path):
                gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
                valid = gt != IGNORE_INDEX
                gt_valid = gt[valid].astype(np.int64)
                pred_valid = pred[valid].astype(np.int64)
                conf += np.bincount(
                    gt_valid * NUM_CLASSES + pred_valid,
                    minlength=NUM_CLASSES * NUM_CLASSES).reshape(
                        NUM_CLASSES, NUM_CLASSES)
                num_gt += 1

        if (i + 1) % 50 == 0 or i + 1 == len(img_paths):
            print(f'[eval] {i + 1}/{len(img_paths)}')

    avg_ms = float(np.mean(times) * 1000)
    meta = {
        'model': os.path.abspath(args.model),
        'img_dir': os.path.abspath(img_dir),
        'gt_dir': os.path.abspath(gt_dir) if gt_dir else None,
        'input_size': list(size),
        'providers': sess.get_providers(),
        'images': len(img_paths),
        'images_with_gt': num_gt,
        'avg_infer_ms': avg_ms
    }

    if num_gt > 0:
        iou, acc, miou, aacc = compute_metrics(conf)
        print_metrics(iou, acc, miou, aacc, len(img_paths), avg_ms,
                      args.ref_miou)
        meta.update({'mIoU': miou, 'aAcc': aacc})
    else:
        print(f'[result] no GT available, metrics skipped '
              f'images={len(img_paths)} avg_infer={avg_ms:.1f}ms')

    if out_dir:
        with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        if num_gt > 0:
            dump_metrics(out_dir, conf, iou, acc, miou, aacc)
        print(f'[dump] structured outputs saved to {out_dir}')
    if args.save_vis:
        print(f'[visualize] saved {saved} images to {args.vis_dir}')


if __name__ == '__main__':
    main()
