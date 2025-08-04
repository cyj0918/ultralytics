#!/usr/bin/env python3
"""
yolo_predict_stream_eval.py

Run YOLO inference using Ultralytics built-in predict mode with streaming,
measure timing and system metrics, and compute ground-truth metrics by reading
predict and ground-truth TXT files. Output a comprehensive JSON summary.
"""

import argparse
import sys
import time
import torch
import numpy as np
import psutil
import json
import yaml
from pathlib import Path
from datetime import datetime
import thop
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO streaming inference + metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Path to model weights (.pt)")
    parser.add_argument("--source", "-s", type=str, required=True,
                        help="Image/video directory for inference")
    parser.add_argument("--dataset-config", "-d", type=str, default=None,
                        help="YAML dataset config for GT metrics (unused for path)")
    parser.add_argument("--conf", "-c", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="IoU threshold for NMS and GT matching")
    parser.add_argument("--imgsz", "--img-size", type=int, default=640,
                        help="Inference image size")
    parser.add_argument("--device", type=str, default="mps",
                        choices=["mps"], help="Compute device")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save annotated results")
    parser.add_argument("--save-txt", action="store_true", default=True,
                        help="Save predictions in YOLO TXT format")
    parser.add_argument("--save-conf", action="store_true", default=True,
                        help="Include confidence in TXT")
    parser.add_argument("--enable-metrics", action="store_true", default=True,
                        help="Compute ground-truth metrics")
    parser.add_argument("--project", type=str, default="runs/detect",
                        help="Ultralytics project folder")
    parser.add_argument("--name", type=str, default="exp",
                        help="Ultralytics run name")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    return parser.parse_args()


def measure_model_info(model_path, device):
    model = YOLO(model_path)
    model.model = model.model.to(device)
    dummy = torch.randn(1, 3, 640, 640).to(device)
    flops, params = thop.profile(model.model, inputs=(dummy,), verbose=False)
    return {
        "model_path": model_path,
        "parameters_M": params / 1e6,
        "flops_G": flops / 1e9,
        "device": str(device),
        "model_size_mb": Path(model_path).stat().st_size / (1024 * 1024),
        "timestamp": datetime.now().isoformat()
    }


def measure_system_metrics(device):
    mem = psutil.virtual_memory()
    mem_mb = (mem.total - mem.available) / (1024 * 1024)
    mps_mb = 0
    if device.type == "mps":
        try:
            mps_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
        except:
            mps_mb = mem_mb
    return mem_mb, mps_mb


def load_txt(path):
    cls_list, boxes, confs = [], [], []
    for line in open(path):
        parts = line.split()
        cls_list.append(int(parts[0]))
        box = list(map(float, parts[1:5]))
        boxes.append(box)
        confs.append(float(parts[5]) if len(parts) > 5 else 1.0)
    return cls_list, boxes, confs


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x - w/2, y - h/2, x + w/2, y + h/2]


def compute_iou(a, b):
    xa1, ya1, xa2, ya2 = xywh_to_xyxy(a)
    xb1, yb1, xb2, yb2 = xywh_to_xyxy(b)
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter/union if union > 0 else 0.0


def load_gt(txt_path):
    if not Path(txt_path).exists():
        return [], []
    cls, boxes, _ = load_txt(txt_path)
    return cls, boxes


def evaluate_gt(pred_dir, gt_dir, iou_thres):
    pfs = sorted(Path(pred_dir).glob("*.txt"))
    gfs = sorted(Path(gt_dir).glob("*.txt"))
    tp = fp = fn = 0
    for pf, gf in zip(pfs, gfs):
        pcls, pboxes, _ = load_txt(pf)
        gcls, gboxes    = load_gt(gf)
        matched = set()
        for c, box in zip(pcls, pboxes):
            ious = [compute_iou(box, gb) for gc, gb in zip(gcls, gboxes) if gc == c]
            if ious and max(ious) >= iou_thres:
                idx = ious.index(max(ious))
                if idx not in matched:
                    tp += 1
                    matched.add(idx)
                else:
                    fp += 1
            else:
                fp += 1
        fn += len(gboxes) - len(matched)
    prec = tp/(tp+fp) if tp+fp > 0 else 0.0
    rec  = tp/(tp+fn) if tp+fn > 0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0.0
    return prec, rec, f1


def main():
    args = parse_args()
    device = (torch.device(args.device)
              if args.device == "mps" and torch.backends.mps.is_available()
              else torch.device("cpu"))

    model_info = measure_model_info(args.model, device)
    model = YOLO(args.model)

    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    lat, pre, post, mem, mps = [], [], [], [], []
    print("Running streaming inference...")
    start_all = time.perf_counter()
    for res in model(
        args.source,
        conf=args.conf,
        iou=args.iou,
        device=device,
        imgsz=args.imgsz,
        save=args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=args.project,
        name=args.name,
        verbose=False,
        stream=True
    ):
        s = res.speed
        pre.append(s.get("preprocess", 0))
        lat.append(s.get("inference", 0))
        post.append(s.get("postprocess", 0))
        mu, mm = measure_system_metrics(device)
        mem.append(mu)
        mps.append(mm)
        total += 1
    end_all = time.perf_counter()

    tm = {
        "total_images": total,
        "avg_inference_time_ms": float(np.mean(lat)) if lat else 0.0,
        "std_inference_time_ms": float(np.std(lat)) if lat else 0.0,
        "median_inference_time_ms": float(np.median(lat)) if lat else 0.0,
        "min_inference_time_ms": float(np.min(lat)) if lat else 0.0,
        "max_inference_time_ms": float(np.max(lat)) if lat else 0.0,
        "fps": (1000/np.mean(lat)) if lat else 0.0,
        "avg_preprocess_time_ms": float(np.mean(pre)) if pre else 0.0,
        "avg_postprocess_time_ms": float(np.mean(post)) if post else 0.0,
        "total_time_ms": (end_all - start_all) * 1000
    }

    sm = {
        "avg_memory_usage_mb": float(np.mean(mem)) if mem else 0.0,
        "max_memory_usage_mb": float(np.max(mem)) if mem else 0.0,
        "avg_mps_memory_mb": float(np.mean(mps)) if mps else 0.0,
        "max_mps_memory_mb": float(np.max(mps)) if mps else 0.0
    }

    # Derive GT directory by replacing 'images' with 'labels'
    src_parts = Path(args.source).parts
    if "images" in src_parts:
        idx = src_parts.index("images")
        gt_dir = Path(*src_parts[:idx], "labels", *src_parts[idx+1:])
    else:
        gt_dir = None

    pred_dir = save_dir / "labels"
    prec = rec = f1 = 0.0
    if args.enable_metrics and gt_dir and gt_dir.exists():
        prec, rec, f1 = evaluate_gt(pred_dir, gt_dir, args.iou)
    else:
        print(f"Warning: GT directory not found: {gt_dir}")

    output = {
        "model_info": model_info,
        "timing_metrics": tm,
        "system_metrics": sm,
        "detailed_results": []
    }

    out_file = save_dir / "evaluation_summary.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved summary to {out_file}")

    print("\n=== Performance Summary ===")
    print(f"Images processed: {total}")
    print(f"Avg inference: {tm['avg_inference_time_ms']:.2f} ms, FPS: {tm['fps']:.2f}")


if __name__ == "__main__":
    main()
