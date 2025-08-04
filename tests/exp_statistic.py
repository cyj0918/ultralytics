#!/usr/bin/env python3
"""
exp_statistic.py

Evaluate YOLO TXT predictions vs ground truth (TXT only),
compute detection and localization metrics, print and save as JSON.
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO TXT predictions vs ground truth (TXT only)"
    )
    parser.add_argument("--gt-dir", required=True,
                        help="Folder with ground-truth YOLO TXT files")
    parser.add_argument("--pred-dir", required=True,
                        help="Folder with prediction YOLO TXT files")
    parser.add_argument("--iou-thres", type=float, default=0.7,
                        help="IoU threshold for matching TP/FP")
    parser.add_argument("--output", "-o", type=str, default="statistics.json",
                        help="Output JSON file path")
    return parser.parse_args()

def load_txt(path):
    cls, boxes = [], []
    for line in open(path):
        parts = line.strip().split()
        cls.append(int(parts[0]))
        boxes.append(list(map(float, parts[1:5])))
    return cls, boxes

def xywh_to_xyxy_norm(box):
    x,y,w,h = box
    return [x-w/2, y-h/2, x+w/2, y+h/2]

def compute_iou(a, b):
    xa1,ya1,xa2,ya2 = a
    xb1,yb1,xb2,yb2 = b
    xi1, yi1 = max(xa1,xb1), max(ya1,yb1)
    xi2, yi2 = min(xa2,xb2), min(ya2,yb2)
    inter = max(0, xi2-xi1)*max(0, yi2-yi1)
    area_a = max(0, xa2-xa1)*max(0, ya2-ya1)
    area_b = max(0, xb2-xb1)*max(0, yb2-yb1)
    union = area_a + area_b - inter
    return inter/union if union>0 else 0.0

def evaluate_detection(gt_dir, pred_dir, iou_thres):
    gt_files = sorted(Path(gt_dir).glob("*.txt"))
    pred_files = sorted(Path(pred_dir).glob("*.txt"))
    tp = fp = fn = 0
    for gf, pf in zip(gt_files, pred_files):
        gcls, gboxes = load_txt(gf)
        pcls, pboxes = load_txt(pf)
        gxy = [xywh_to_xyxy_norm(b) for b in gboxes]
        matched = set()
        for pc, pb in zip(pcls, pboxes):
            pxy = xywh_to_xyxy_norm(pb)
            # same-class IoUs
            ious = [(i, compute_iou(pxy, gxy[i]))
                    for i, gc in enumerate(gcls) if gc == pc]
            if ious and max(iou for _, iou in ious) >= iou_thres:
                idx, _ = max(ious, key=lambda x: x[1])
                if idx not in matched:
                    tp += 1
                    matched.add(idx)
                else:
                    fp += 1
            else:
                fp += 1
        fn += len(gxy) - len(matched)
    prec = tp/(tp+fp) if tp+fp>0 else 0.0
    rec  = tp/(tp+fn) if tp+fn>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
    return prec, rec, f1

def evaluate_localization(gt_dir, pred_dir, iou_thres):
    gt_files = sorted(Path(gt_dir).glob("*.txt"))
    pred_files = sorted(Path(pred_dir).glob("*.txt"))
    class_stats = {}
    for gf, pf in zip(gt_files, pred_files):
        gcls, gboxes = load_txt(gf)
        pcls, pboxes = load_txt(pf)
        gxy = [xywh_to_xyxy_norm(b) for b in gboxes]
        for pc, pb in zip(pcls, pboxes):
            pxy = xywh_to_xyxy_norm(pb)
            # match same class
            stats = class_stats.setdefault(pc, {"ious_tp": [], "mses": []})
            ious = [(i, compute_iou(pxy, gxy[i]))
                    for i, gc in enumerate(gcls) if gc == pc]
            if ious and max(iou for _, iou in ious) >= iou_thres:
                idx, best_iou = max(ious, key=lambda x: x[1])
                stats["ious_tp"].append(best_iou)
                # MSE
                gb = gxy[idx]
                mse = np.mean((np.array(pxy) - np.array(gb))**2)
                stats["mses"].append(mse)
    loc_table = []
    for cid, s in class_stats.items():
        ious = np.array(s["ious_tp"]) if s["ious_tp"] else np.array([0.0])
        mses = np.array(s["mses"]) if s["mses"] else np.array([0.0])
        avg_iou = float(ious.mean())
        std_iou = float(ious.std())
        dice = float((2*ious/(1+ious)).mean())
        mse = float(mses.mean())
        loc_table.append({
            "Class_ID": cid,
            "Avg_IoU_TP": avg_iou,
            "IoU_Std": std_iou,
            "Dice_Coefficient": dice,
            "Box_MSE": mse
        })
    return loc_table

def main():
    args = parse_args()
    prec, rec, f1 = evaluate_detection(args.gt_dir, args.pred_dir, args.iou_thres)
    loc_table = evaluate_localization(args.gt_dir, args.pred_dir, args.iou_thres)

    output = {
        "performance_metrics": {
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1
        },
        "localization_table": loc_table
    }

    # Print to stdout
    print(json.dumps(output, indent=2))

    # Save to file
    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved statistics to {out_path}")

if __name__ == "__main__":
    main()