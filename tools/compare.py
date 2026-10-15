# Copyright (c) OpenMMLab. All rights reserved.
"""Standalone prediction-vs-GT comparison with official Cityscapes metrics.

Compares structured prediction dumps (e.g. from ``eval_onnx.py`` or
``eval_axera.py``) against ``*_gtFine_labelTrainIds.png`` ground truth and
reports the official metrics: 19-class IoU/Acc, mIoU, aAcc
(ignore_index=255). Optionally saves visual diffs (GT color | prediction
color | error map).

    <pred-dir>/<city>/<stem>_pred.png   vs
    <gt-dir>/<city>/<stem>_gtFine_labelTrainIds.png

Outputs:
    <out-dir>/metrics.json
    <out-dir>/metrics.csv
    <out-dir>/confusion_matrix.csv
    <out-dir>/diff/<city>/<stem>_diff.png   (with --save-diff)

Example:
    python compare.py \
        --pred-dir axera_pred-u16/pred \
        --gt-dir ../dataset/cityscapes/gtFine/val \
        --out-dir compare/axera_pred-u16_vs_gt \
        --ref-miou 79.99 --save-diff 5
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

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

NUM_CLASSES = 19
IGNORE_INDEX = 255


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare predictions with Cityscapes GT')
    parser.add_argument(
        '--pred-dir',
        required=True,
        help='dir with predicted label id pngs, e.g. '
        'runs/onnx_pred/<model>/pred')
    parser.add_argument(
        '--gt-dir',
        required=True,
        help='dir with *_gtFine_labelTrainIds.png ground truth')
    parser.add_argument(
        '--pred-suffix',
        default='_pred.png',
        help='suffix of prediction files')
    parser.add_argument(
        '--gt-suffix',
        default='_gtFine_labelTrainIds.png',
        help='suffix of GT files')
    parser.add_argument(
        '--out-dir',
        default=None,
        help='output dir; defaults to compare/<pred-stem>_vs_gt')
    parser.add_argument(
        '--save-diff',
        type=int,
        default=0,
        help='save visual diff (GT|pred|error) for the first N pairs')
    parser.add_argument(
        '--ref-miou',
        type=float,
        default=None,
        help='optional reference mIoU to show the difference')
    return parser.parse_args()


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
    aacc = float(intersection.sum() / total * 100) if total > 0 else float(
        'nan')
    return iou, acc, miou, aacc


def make_diff_image(img_shape, pred, gt, palette):
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:len(palette)] = palette
    gt_safe = np.where(gt == IGNORE_INDEX, 0, gt).astype(np.uint8)
    gt_color = lut[gt_safe]
    pred_color = lut[pred]
    error = np.zeros((*img_shape, 3), dtype=np.uint8)
    valid = gt != IGNORE_INDEX
    correct = (pred == gt) & valid
    wrong = (~correct) & valid
    error[correct] = (0, 255, 0)
    error[wrong] = (0, 0, 255)
    return np.concatenate([gt_color, pred_color, error], axis=1)


def main():
    args = parse_args()

    if args.out_dir is None:
        stem = os.path.basename(os.path.normpath(args.pred_dir))
        if stem == 'pred':
            stem = os.path.basename(
                os.path.dirname(os.path.normpath(args.pred_dir)))
        args.out_dir = os.path.join('compare', f'{stem}_vs_gt')
    os.makedirs(args.out_dir, exist_ok=True)

    pred_paths = sorted(
        glob.glob(
            os.path.join(args.pred_dir, '**', f'*{args.pred_suffix}'),
            recursive=True))
    assert pred_paths, (
        f'no prediction files matching *{args.pred_suffix} under '
        f'{args.pred_dir}')
    print(f'[compare] pred_dir={args.pred_dir}')
    print(f'[compare] gt_dir={args.gt_dir} pairs={len(pred_paths)}')

    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    missing = 0
    saved = 0
    for pred_path in pred_paths:
        rel = os.path.relpath(pred_path, args.pred_dir)
        rel_parent = os.path.dirname(rel)
        stem = os.path.basename(pred_path)[:-len(args.pred_suffix)]
        gt_path = os.path.join(args.gt_dir, rel_parent,
                               f'{stem}{args.gt_suffix}')
        if not os.path.isfile(gt_path):
            missing += 1
            print(f'[warn] missing GT for {rel}')
            continue

        pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        assert pred is not None, f'failed to read {pred_path}'
        assert gt is not None, f'failed to read {gt_path}'
        assert pred.shape == gt.shape, (
            f'shape mismatch {pred_path} {pred.shape} vs {gt_path} {gt.shape}')

        valid = gt != IGNORE_INDEX
        gt_valid = gt[valid].astype(np.int64)
        pred_valid = pred[valid].astype(np.int64)
        conf += np.bincount(
            gt_valid * NUM_CLASSES + pred_valid,
            minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES,
                                                         NUM_CLASSES)

        if saved < args.save_diff:
            diff = make_diff_image(gt.shape, pred, gt, CITYSCAPES_PALETTE)
            diff_path = os.path.join(args.out_dir, 'diff', rel_parent,
                                     f'{stem}_diff.png')
            os.makedirs(os.path.dirname(diff_path), exist_ok=True)
            cv2.imwrite(diff_path, diff)
            saved += 1

    iou, acc, miou, aacc = compute_metrics(conf)
    print(f'{"class":<15s} {"IoU":>7s} {"Acc":>7s}')
    for cid, name in enumerate(CITYSCAPES_CLASSES):
        print(f'{name:<15s} {iou[cid]:7.2f} {acc[cid]:7.2f}')
    print(f'[result] mIoU={miou:.2f} aAcc={aacc:.2f} '
          f'pairs={len(pred_paths) - missing} missing_gt={missing}')
    if args.ref_miou is not None:
        print(f'[result] reference mIoU={args.ref_miou:.2f} '
              f'diff={miou - args.ref_miou:+.2f}')

    metrics = {
        'pred_dir': os.path.abspath(args.pred_dir),
        'gt_dir': os.path.abspath(args.gt_dir),
        'pairs': len(pred_paths) - missing,
        'missing_gt': missing,
        'mIoU': miou,
        'aAcc': aacc,
        'ref_miou': args.ref_miou,
        'mIoU_diff_vs_ref': (miou - args.ref_miou)
        if args.ref_miou is not None else None,
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
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out_dir, 'metrics.csv'), 'w') as f:
        f.write('class,iou,acc\n')
        for cid, name in enumerate(CITYSCAPES_CLASSES):
            f.write(f'{name},{iou[cid]:.2f},{acc[cid]:.2f}\n')
        f.write(f'mIoU,{miou:.2f},\n')
        f.write(f'aAcc,,{aacc:.2f}\n')
    np.savetxt(
        os.path.join(args.out_dir, 'confusion_matrix.csv'),
        conf,
        fmt='%d',
        delimiter=',',
        header=','.join(CITYSCAPES_CLASSES),
        comments='')
    print(f'[dump] comparison results saved to {args.out_dir}')
    if args.save_diff:
        print(f'[visualize] saved {saved} diff images to '
              f'{os.path.join(args.out_dir, "diff")}')


if __name__ == '__main__':
    main()
