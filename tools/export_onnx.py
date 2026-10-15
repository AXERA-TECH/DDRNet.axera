# Copyright (c) OpenMMLab. All rights reserved.
"""Export DDRNet (Cityscapes, OpenMMLab open-source weights) to ONNX.

Input : float32 RGB NCHW with fixed shape (1, 3, H, W), already normalized
        with mean/std, i.e. ``(rgb - mean) / std`` done in the caller
        preprocessing (BGR->RGB, mean/std are NOT part of the graph)
Output: segmentation logits (1, num_classes, H, W)
Shape is fixed to ``--input-size`` (default 1024 1024); the exported file is
named ``<name>_<W>x<H>.onnx``. For Cityscapes official evaluation use
``--input-size 1024 2048`` (H W, i.e. full-resolution 2048x1024 input).

Weights are expected under ``weights/`` (downloaded automatically if absent),
the exported ONNX models are written to ``runs/``.

Usage:
    python tools/deployment/export_onnx.py
"""
import argparse
import os
import sys
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn as nn

PROJECT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_DIR = os.path.join(PROJECT, 'weights')
sys.path.insert(0, PROJECT)

from mmengine.config import Config  # noqa: E402
from mmengine.runner import load_checkpoint  # noqa: E402
from mmseg.registry import MODELS  # noqa: E402
from mmseg.utils import register_all_modules  # noqa: E402

CKPT_ROOT = 'https://download.openmmlab.com/mmsegmentation/v0.5/ddrnet'

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

MODELS_CONFIG = {
    'ddrnet23': dict(
        cfg='configs/ddrnet/'
        'ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024.py',
        ckpt='ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024'
        '_20230425_162633-81601db0.pth',
        url=f'{CKPT_ROOT}/ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024/'
        'ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024'
        '_20230425_162633-81601db0.pth',
        out='runs/ddrnet23_cityscapes-{w}x{h}.onnx'),
    'ddrnet23-slim': dict(
        cfg='configs/ddrnet/'
        'ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024.py',
        ckpt='ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024'
        '_20230426_145312-6a5e5174.pth',
        url=f'{CKPT_ROOT}/ddrnet_23-slim_in1k-pre_2xb6-120k'
        '-cityscapes-1024x1024/ddrnet_23-slim_in1k-pre_2xb6-120k'
        '-cityscapes-1024x1024_20230426_145312-6a5e5174.pth',
        out='runs/ddrnet23-slim_cityscapes-{w}x{h}.onnx'),
}


def download_if_missing(url: str, path: str) -> str:
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f'[download] {url} -> {path}')
        tmp_path = path + '.tmp'
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, path)
    return path


def convert_syncbn(module: nn.Module) -> None:
    """Replace SyncBatchNorm with BatchNorm2d (identical state_dict keys)."""
    for name, child in module.named_children():
        if isinstance(child, nn.SyncBatchNorm):
            bn = nn.BatchNorm2d(
                child.num_features,
                child.eps,
                child.momentum,
                child.affine,
                child.track_running_stats)
            with torch.no_grad():
                bn.load_state_dict(child.state_dict())
            setattr(module, name, bn)
        else:
            convert_syncbn(child)


class ExportModel(nn.Module):
    """Wrap backbone + decode head (+ logits resize) for ONNX export.

    Preprocessing (BGR->RGB, mean/std normalization) is intentionally kept
    out of the graph and must be done by the caller.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.model.backbone(x)
        seg_logits = self.model.decode_head.predict(
            feat, [dict(img_shape=x.shape[2:])], self.model.test_cfg)
        return seg_logits


def build_model(cfg_path: str, ckpt_path: str) -> ExportModel:
    cfg = Config.fromfile(cfg_path)
    cfg.model.backbone.pop('init_cfg', None)
    model = MODELS.build(cfg.model)
    convert_syncbn(model)
    load_checkpoint(model, ckpt_path, map_location='cpu')
    model.eval()
    return ExportModel(model).eval()


def export(model: ExportModel, out_path: str, opset: int, device: str,
           input_size):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    dummy = torch.randn(1, 3, input_size[0], input_size[1], device=device)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            out_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=opset,
            do_constant_folding=True)
    print(f'[export] saved {out_path}')


def get_sample(path: str, size):
    if path and os.path.isfile(path):
        img = cv2.imread(path)
    else:
        img = np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8)
    return cv2.resize(img, (size[1], size[0]))


def preprocess(img_bgr, size):
    img = cv2.resize(
        img_bgr, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)[:, :, ::-1]
    img = (img - MEAN) / STD
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None])


def verify(model: ExportModel, onnx_path: str, input_size):
    """Compare ONNX Runtime (CPU) against PyTorch (CPU) reference."""
    import onnxruntime as ort

    ref = model.cpu().eval()
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    img_path = os.path.join(PROJECT, 'demo/classroom__rgb_00283.jpg')
    for h, w in [tuple(input_size)]:
        img = get_sample(img_path, (h, w))
        inp = preprocess(img, (h, w))
        with torch.no_grad():
            torch_out = ref(torch.from_numpy(inp)).numpy()
        ort_out = sess.run(['output'], {'input': inp})[0]
        diff = np.abs(torch_out - ort_out)
        mismatch = (torch_out.argmax(1) != ort_out.argmax(1)).sum()
        print(f'[verify] shape={inp.shape} max_abs_diff={diff.max():.3e} '
              f'mean_abs_diff={diff.mean():.3e} '
              f'argmax_mismatch={mismatch}/{diff[0, 0].size}')
        assert diff.shape == (1, torch_out.shape[1], h, w)
        assert diff.max() < 1e-4, 'onnx output mismatch'
        assert mismatch / diff[0, 0].size < 1e-4, 'too many argmax mismatches'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=list(MODELS_CONFIG))
    parser.add_argument('--opset', type=int, default=11)
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--input-size', type=int, nargs=2, default=[1024, 1024])
    parser.add_argument('--skip-verify', action='store_true')
    args = parser.parse_args()

    register_all_modules()
    for name in args.models:
        meta = MODELS_CONFIG[name]
        cfg_path = os.path.join(PROJECT, meta['cfg'])
        ckpt_path = download_if_missing(
            meta['url'], os.path.join(WEIGHTS_DIR, meta['ckpt']))
        out_path = os.path.join(
            PROJECT,
            meta['out'].format(
                w=args.input_size[1], h=args.input_size[0]))
        print(f'[build] {name}')
        model = build_model(cfg_path, ckpt_path)
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        export(model.to(device), out_path, args.opset, device, args.input_size)
        if not args.skip_verify:
            verify(model, out_path, args.input_size)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
