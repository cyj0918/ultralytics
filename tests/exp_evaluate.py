# Train: yolo detect train data=ultralytics/cfg/datasets/mf.yaml model=ultralytics/cfg/models/modified/yolo11.yaml epochs=100 batch=16 lr0=0.01 optimizer=SGD device=mps  plots=True save=True project=exp0 name=exp0
# Test: python3 tests/exp_evaluate.py --dataset ultralytics/cfg/datasets/mf.yaml --output-dir results --runs 3 --conf-thresh 0.25 --iou-thresh 0.7 --img-size 640

# Single Model Test with Ground Truth Comparison

import json
import time
import torch
import numpy as np
import cv2
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.utils import ops
import thop
import argparse
import yaml
import pandas as pd
from typing import Dict, List, Tuple, Optional


class YOLOv11SingleModelEvaluator:
    def __init__(self, model_path: str, model_name: str, dataset_config: str, 
                 conf_thresh: float = 0.25, iou_thresh: float = 0.7, 
                 img_size: int = 640, device: str = 'mps'):
        """
        初始化單一模型評估器
        
        Args:
            model_path: 模型權重檔案路徑或yaml配置檔案路徑
            model_name: 模型名稱
            dataset_config: 數據集配置檔案路徑
            conf_thresh: 置信度閾值
            iou_thresh: IoU閾值
            img_size: 輸入圖片尺寸
            device: 運算設備
        """
        self.model_path = model_path
        self.model_name = model_name
        self.dataset_config = dataset_config
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size if isinstance(img_size, (list, tuple)) else [img_size, img_size]
        
        # 設置設備
        if device == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        print(f"Using device: {self.device}")
        print(f"Loading model: {model_path}")
        
        # 初始化模型
        self.model = YOLO(model_path).to(self.device)
        self.model.fuse()
        
        # 載入數據集配置
        self.dataset_info = self._load_dataset_config()
        
        # 初始化結果儲存
        self.results = {
            'model_info': self._get_model_info(),
            'dataset_info': self.dataset_info,
            'evaluation_metrics': {},
            'detailed_comparison': [],
            'class_performance': {},
            'error_analysis': {
                'false_positives': [],
                'false_negatives': [],
                'misclassifications': []
            }
        }
    
    def _load_dataset_config(self) -> Dict:
        """載入數據集配置檔案"""
        with open(self.dataset_config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _get_model_info(self) -> Dict:
        """獲取模型資訊"""
        # 計算模型參數和FLOPs
        dummy_input = torch.randn(1, 3, *self.img_size).to(self.device)
        try:
            flops, params = thop.profile(self.model.model, inputs=(dummy_input,), verbose=False)
            flops = flops / 1e9  # GFLOPs
            params = params / 1e6  # Millions
        except Exception as e:
            print(f"Warning: Could not calculate FLOPs and params: {e}")
            flops, params = 0, 0
        
        return {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'parameters_M': params,
            'flops_G': flops,
            'input_size': self.img_size,
            'confidence_threshold': self.conf_thresh,
            'iou_threshold': self.iou_thresh,
            'device': str(self.device),
            'timestamp': datetime.now().isoformat()
        }
    
    def _synchronize_device(self):
        """設備同步"""
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elif self.device.type == 'mps':
            pass  # MPS不需要顯式同步
    
    def _measure_inference_time(self, test_images: List[str], num_samples: int = 100) -> Dict:
        """測量推理時間"""
        print(f"Measuring inference time on {min(num_samples, len(test_images))} images...")
        
        latencies = []
        memory_usage = []
        
        # 預熱
        dummy_input = torch.randn(1, 3, *self.img_size).to(self.device)
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        self._synchronize_device()
        
        # 實際測量
        for i, img_path in enumerate(test_images[:num_samples]):
            if not os.path.exists(img_path):
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # 測量記憶體使用（如果是CUDA）
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            elif self.device.type == 'mps':
                pass  # MPS記憶體測量較為複雜
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                results = self.model(img_path, conf=self.conf_thresh, iou=self.iou_thresh, 
                                   imgsz=self.img_size, verbose=False)
            
            self._synchronize_device()
            
            latency = (time.perf_counter() - start_time) * 1000  # 轉換為毫秒
            latencies.append(latency)
            
            # 記憶體使用
            if self.device.type == 'cuda':
                memory_usage.append(torch.cuda.max_memory_allocated() / (1024 ** 2))  # MB
            
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{min(num_samples, len(test_images))} images")
        
        return {
            'avg_inference_time_ms': np.mean(latencies),
            'std_inference_time_ms': np.std(latencies),
            'median_inference_time_ms': np.median(latencies),
            'min_inference_time_ms': np.min(latencies),
            'max_inference_time_ms': np.max(latencies),
            'fps': 1000 / np.mean(latencies),
            'avg_memory_usage_mb': np.mean(memory_usage) if memory_usage else 0
        }
    
    def _load_ground_truth_labels(self, label_path: str) -> List[Dict]:
        """載入真實標註"""
        if not os.path.exists(label_path):
            return []
        
        labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        labels.append({
                            'class_id': class_id,
                            'bbox': [x_center, y_center, width, height],
                            'bbox_format': 'xywh_normalized'
                        })
        return labels
    
    def _convert_bbox_format(self, bbox: List[float], from_format: str, 
                            to_format: str, img_shape: Tuple[int, int]) -> List[float]:
        """轉換邊界框格式"""
        h, w = img_shape[:2]
        
        if from_format == 'xywh_normalized' and to_format == 'xyxy_pixels':
            x_center, y_center, width, height = bbox
            x1 = (x_center - width/2) * w
            y1 = (y_center - height/2) * h
            x2 = (x_center + width/2) * w
            y2 = (y_center + height/2) * h
            return [x1, y1, x2, y2]
        
        elif from_format == 'xyxy_pixels' and to_format == 'xywh_normalized':
            x1, y1, x2, y2 = bbox
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            return [x_center, y_center, width, height]
        
        return bbox
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """計算IoU"""
        # 假設輸入為xyxy格式
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 計算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compare_predictions_with_gt(self, pred_results, gt_labels: List[Dict], 
                                    img_shape: Tuple[int, int], img_path: str) -> Dict:
        """比對預測結果與真實標註"""
        comparison = {
            'image_path': img_path,
            'image_shape': img_shape,
            'ground_truth_count': len(gt_labels),
            'prediction_count': 0,
            'matches': [],
            'false_positives': [],
            'false_negatives': [],
            'true_positives': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        
        if not pred_results or len(pred_results[0].boxes) == 0:
            comparison['prediction_count'] = 0
            comparison['false_negatives'] = gt_labels.copy()
            comparison['recall'] = 0.0
            comparison['precision'] = 0.0 if len(gt_labels) == 0 else 0.0
            return comparison
        
        # 提取預測結果
        predictions = []
        boxes = pred_results[0].boxes
        comparison['prediction_count'] = len(boxes)
        
        for i in range(len(boxes)):
            pred_box = boxes.xyxy[i].cpu().numpy().tolist()
            pred_class = int(boxes.cls[i].cpu().numpy())
            pred_conf = float(boxes.conf[i].cpu().numpy())
            
            predictions.append({
                'bbox': pred_box,
                'class_id': pred_class,
                'confidence': pred_conf,
                'bbox_format': 'xyxy_pixels'
            })
        
        # 將GT標註轉換為像素格式
        gt_boxes_pixel = []
        for gt_label in gt_labels:
            gt_box_pixel = self._convert_bbox_format(
                gt_label['bbox'], 'xywh_normalized', 'xyxy_pixels', img_shape
            )
            gt_boxes_pixel.append({
                'bbox': gt_box_pixel,
                'class_id': gt_label['class_id'],
                'matched': False
            })
        
        # 匹配預測與真實標註
        pred_matched = [False] * len(predictions)
        
        for pred_idx, pred in enumerate(predictions):
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_boxes_pixel):
                if gt['matched']:
                    continue
                
                # 類別必須匹配
                if pred['class_id'] != gt['class_id']:
                    continue
                
                iou = self._calculate_iou(pred['bbox'], gt['bbox'])
                
                if iou > best_iou and iou >= self.iou_thresh:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                # 找到匹配
                pred_matched[pred_idx] = True
                gt_boxes_pixel[best_gt_idx]['matched'] = True
                
                comparison['matches'].append({
                    'prediction': pred,
                    'ground_truth': gt_labels[best_gt_idx],
                    'iou': best_iou,
                    'class_correct': True
                })
        
        # 統計結果
        comparison['true_positives'] = sum(pred_matched)
        
        # False Positives: 未匹配的預測
        for i, pred in enumerate(predictions):
            if not pred_matched[i]:
                comparison['false_positives'].append(pred)
        
        # False Negatives: 未匹配的真實標註
        for i, gt in enumerate(gt_boxes_pixel):
            if not gt['matched']:
                comparison['false_negatives'].append(gt_labels[i])
        
        # 計算指標
        tp = comparison['true_positives']
        fp = len(comparison['false_positives'])
        fn = len(comparison['false_negatives'])
        
        comparison['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        comparison['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        comparison['f1_score'] = (2 * comparison['precision'] * comparison['recall'] / 
                                 (comparison['precision'] + comparison['recall'])) if (comparison['precision'] + comparison['recall']) > 0 else 0.0
        
        return comparison
    
    def evaluate_model(self, output_dir: str = None) -> Dict:
        """評估模型性能"""
        if output_dir is None:
            output_dir = f"evaluation_results_{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting evaluation for model: {self.model_name}")
        print(f"Results will be saved to: {output_path}")
        
        # 1. 運行標準驗證獲取整體指標
        print("\n1. Running standard validation...")
        val_results = self.model.val(
            data=self.dataset_config,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=str(self.device),
            imgsz=self.img_size,
            save_json=True,
            save_hybrid=True,
            plots=True,
            project=str(output_path),
            name='validation'
        )
        
        # 提取標準指標
        self.results['evaluation_metrics'] = {
            'mAP_0.5': float(val_results.box.map50),
            'mAP_0.5:0.95': float(val_results.box.map),
            'precision': float(val_results.box.mp),
            'recall': float(val_results.box.mr),
            'f1_score': float(2 * val_results.box.mp * val_results.box.mr / (val_results.box.mp + val_results.box.mr)) if (val_results.box.mp + val_results.box.mr) > 0 else 0.0
        }
        
        # 2. 獲取測試圖片路徑
        test_images = self._get_test_images()
        print(f"\n2. Found {len(test_images)} test images")
        
        # 3. 測量推理時間
        print("\n3. Measuring inference time...")
        timing_results = self._measure_inference_time(test_images)
        self.results['evaluation_metrics'].update(timing_results)
        
        # 4. 詳細比對預測與標註
        print("\n4. Comparing predictions with ground truth annotations...")
        detailed_comparisons = []
        class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0, 'total_pred': 0})
        
        for i, img_path in enumerate(test_images):
            if i % 50 == 0:
                print(f"Processing image {i+1}/{len(test_images)}: {Path(img_path).name}")
            
            # 獲取預測結果
            pred_results = self.model(img_path, conf=self.conf_thresh, iou=self.iou_thresh, 
                                    imgsz=self.img_size, verbose=False)
            
            # 載入對應的標註檔案
            label_path = self._get_label_path(img_path)
            gt_labels = self._load_ground_truth_labels(label_path)
            
            # 獲取圖片尺寸
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_shape = img.shape
            
            # 比對結果
            comparison = self._compare_predictions_with_gt(pred_results, gt_labels, img_shape, img_path)
            detailed_comparisons.append(comparison)
            
            # 更新類別統計
            for gt_label in gt_labels:
                class_id = gt_label['class_id']
                class_stats[class_id]['total_gt'] += 1
            
            if pred_results and len(pred_results[0].boxes) > 0:
                for box in pred_results[0].boxes:
                    class_id = int(box.cls.cpu().numpy())
                    class_stats[class_id]['total_pred'] += 1
            
            # 統計TP, FP, FN
            class_stats_update = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
            
            for match in comparison['matches']:
                class_id = match['ground_truth']['class_id']
                class_stats_update[class_id]['tp'] += 1
            
            for fp in comparison['false_positives']:
                class_id = fp['class_id']
                class_stats_update[class_id]['fp'] += 1
            
            for fn in comparison['false_negatives']:
                class_id = fn['class_id']
                class_stats_update[class_id]['fn'] += 1
            
            # 更新總體統計
            for class_id, stats in class_stats_update.items():
                for key, value in stats.items():
                    class_stats[class_id][key] += value
        
        self.results['detailed_comparison'] = detailed_comparisons
        
        # 5. 計算類別級別性能
        print("\n5. Calculating class-level performance...")
        class_names = self.dataset_info.get('names', {})
        
        for class_id, stats in class_stats.items():
            tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            
            class_name = class_names.get(class_id, f'class_{class_id}')
            
            self.results['class_performance'][class_name] = {
                'class_id': class_id,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'total_ground_truth': stats['total_gt'],
                'total_predictions': stats['total_pred'],
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        # 6. 錯誤分析
        print("\n6. Performing error analysis...")
        self._perform_error_analysis(detailed_comparisons)
        
        # 7. 保存結果
        print("\n7. Saving results...")
        self._save_results(output_path)
        
        print(f"\nEvaluation completed! Results saved to: {output_path}")
        return self.results
    
    def _get_test_images(self) -> List[str]:
        """獲取測試圖片路徑列表"""
        dataset_root = Path(self.dataset_info.get('path', '.'))
        
        # 優先使用test路徑，如果沒有則使用val路徑
        test_path = self.dataset_info.get('test', None)
        if test_path is None:
            # 如果yaml中沒有定義test，則嘗試常見的test路徑
            possible_test_paths = ['images/test', 'test', 'images/val', 'val']
            test_images_dir = None
            
            for path in possible_test_paths:
                if not os.path.isabs(path):
                    candidate_dir = dataset_root / path
                else:
                    candidate_dir = Path(path)
                
                if candidate_dir.exists():
                    test_images_dir = candidate_dir
                    print(f"Using test images from: {test_images_dir}")
                    break
        else:
            if not os.path.isabs(test_path):
                test_images_dir = dataset_root / test_path
            else:
                test_images_dir = Path(test_path)
        
        if test_images_dir is None or not test_images_dir.exists():
            raise FileNotFoundError(f"Test images directory not found. Checked paths under: {dataset_root}")
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        test_images = []
        
        for ext in image_extensions:
            test_images.extend(list(test_images_dir.glob(f'*{ext}')))
            test_images.extend(list(test_images_dir.glob(f'*{ext.upper()}')))
        
        print(f"Found {len(test_images)} test images in {test_images_dir}")
        return [str(img) for img in test_images]
    
    def _get_label_path(self, img_path: str) -> str:
        """根據圖片路徑獲取對應的標註檔案路徑"""
        img_path = Path(img_path)
        
        # 獲取數據集根目錄
        dataset_root = Path(self.dataset_info.get('path', '.'))
        
        # 將images路徑替換為labels路徑
        # 處理您的數據集結構: /path/to/dataset/images/test -> /path/to/dataset/labels/test
        img_path_str = str(img_path)
        
        if '/images/' in img_path_str:
            label_path_str = img_path_str.replace('/images/', '/labels/')
        elif '\\images\\' in img_path_str:
            label_path_str = img_path_str.replace('\\images\\', '\\labels\\')
        else:
            # 如果路徑中沒有images，嘗試相對於數據集根目錄構建labels路徑
            rel_path = img_path.relative_to(dataset_root / 'images')
            label_path_str = str(dataset_root / 'labels' / rel_path)
        
        label_path = Path(label_path_str).with_suffix('.txt')
        
        return str(label_path)
    
    def _perform_error_analysis(self, detailed_comparisons: List[Dict]):
        """進行錯誤分析"""
        all_fps = []
        all_fns = []
        all_misclassifications = []
        
        for comparison in detailed_comparisons:
            # 收集False Positives
            for fp in comparison['false_positives']:
                fp_info = fp.copy()
                fp_info['image_path'] = comparison['image_path']
                all_fps.append(fp_info)
            
            # 收集False Negatives
            for fn in comparison['false_negatives']:
                fn_info = fn.copy()
                fn_info['image_path'] = comparison['image_path']
                all_fns.append(fn_info)
        
        # 按置信度排序FP（高置信度的錯誤檢測更值得關注）
        all_fps.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        self.results['error_analysis'] = {
            'false_positives': all_fps[:100],  # 保存前100個
            'false_negatives': all_fns[:100],   # 保存前100個
            'false_positive_summary': {
                'total_count': len(all_fps),
                'avg_confidence': np.mean([fp.get('confidence', 0) for fp in all_fps]) if all_fps else 0,
                'class_distribution': self._get_class_distribution([fp['class_id'] for fp in all_fps])
            },
            'false_negative_summary': {
                'total_count': len(all_fns),
                'class_distribution': self._get_class_distribution([fn['class_id'] for fn in all_fns])
            }
        }
    
    def _get_class_distribution(self, class_ids: List[int]) -> Dict:
        """獲取類別分布"""
        if not class_ids:
            return {}
        
        unique, counts = np.unique(class_ids, return_counts=True)
        class_names = self.dataset_info.get('names', {})
        
        distribution = {}
        for class_id, count in zip(unique, counts):
            class_name = class_names.get(class_id, f'class_{class_id}')
            distribution[class_name] = int(count)
        
        return distribution
    
    def _save_results(self, output_path: Path):
        """保存評估結果"""
        # 保存完整結果為JSON
        results_file = output_path / 'evaluation_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存摘要結果
        summary = {
            'Model': self.model_name,
            'Dataset': self.dataset_config,
            'mAP_0.5': self.results['evaluation_metrics']['mAP_0.5'],
            'mAP_0.5:0.95': self.results['evaluation_metrics']['mAP_0.5:0.95'],
            'Precision': self.results['evaluation_metrics']['precision'],
            'Recall': self.results['evaluation_metrics']['recall'],
            'F1_Score': self.results['evaluation_metrics']['f1_score'],
            'Avg_Inference_Time_ms': self.results['evaluation_metrics']['avg_inference_time_ms'],
            'FPS': self.results['evaluation_metrics']['fps'],
            'Parameters_M': self.results['model_info']['parameters_M'],
            'FLOPs_G': self.results['model_info']['flops_G'],
            'Total_Test_Images': len(self.results['detailed_comparison']),
            'Total_False_Positives': self.results['error_analysis']['false_positive_summary']['total_count'],
            'Total_False_Negatives': self.results['error_analysis']['false_negative_summary']['total_count']
        }
        
        summary_file = output_path / 'evaluation_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 保存類別性能為CSV
        if self.results['class_performance']:
            class_df = pd.DataFrame.from_dict(self.results['class_performance'], orient='index')
            class_df.to_csv(output_path / 'class_performance.csv', index=True)
        
        # 保存詳細比對結果的摘要
        comparison_summary = []
        for comp in self.results['detailed_comparison']:
            comparison_summary.append({
                'image_name': Path(comp['image_path']).name,
                'ground_truth_count': comp['ground_truth_count'],
                'prediction_count': comp['prediction_count'],
                'true_positives': comp['true_positives'],
                'false_positives': len(comp['false_positives']),
                'false_negatives': len(comp['false_negatives']),
                'precision': comp['precision'],
                'recall': comp['recall'],
                'f1_score': comp['f1_score']
            })
        
        if comparison_summary:
            comp_df = pd.DataFrame(comparison_summary)
            comp_df.to_csv(output_path / 'image_by_image_results.csv', index=False)
        
        print(f"Results saved to:")
        print(f"  - Complete results: {results_file}")
        print(f"  - Summary: {summary_file}")
        print(f"  - Class performance: {output_path / 'class_performance.csv'}")
        print(f"  - Image-by-image results: {output_path / 'image_by_image_results.csv'}")


def main():
    parser = argparse.ArgumentParser(description='YOLOv11 Single Model Evaluation with Ground Truth Comparison')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to model weights (.pt) or config (.yaml) file')
    parser.add_argument('--model-name', type=str, required=True,
                       help='Name identifier for the model')
    parser.add_argument('--data', type=str, default='ultralytics/cfg/datasets/mf.yaml',
                       help='Path to dataset configuration file')
    parser.add_argument('--conf-thresh', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou-thresh', type=float, default=0.7,
                       help='IoU threshold')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--device', type=str, default='mps',
                       choices=['mps', 'cuda', 'cpu'], help='Device to use')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # 創建評估器
    evaluator = YOLOv11SingleModelEvaluator(
        model_path=args.model,
        model_name=args.model_name,
        dataset_config=args.data,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        img_size=args.img_size,
        device=args.device
    )
    
    # 執行評估
    results = evaluator.evaluate_model(args.output_dir)
    
    # 打印摘要結果
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.data}")
    print(f"Test Images: {len(results['detailed_comparison'])}")
    print("-"*40)
    print(f"mAP@0.5: {results['evaluation_metrics']['mAP_0.5']:.4f}")
    print(f"mAP@0.5:0.95: {results['evaluation_metrics']['mAP_0.5:0.95']:.4f}")
    print(f"Precision: {results['evaluation_metrics']['precision']:.4f}")
    print(f"Recall: {results['evaluation_metrics']['recall']:.4f}")
    print(f"F1-Score: {results['evaluation_metrics']['f1_score']:.4f}")
    print("-"*40)
    print(f"Avg Inference Time: {results['evaluation_metrics']['avg_inference_time_ms']:.2f} ms")
    print(f"FPS: {results['evaluation_metrics']['fps']:.1f}")
    print(f"Parameters: {results['model_info']['parameters_M']:.2f}M")
    print(f"FLOPs: {results['model_info']['flops_G']:.2f}G")
    print("-"*40)
    print(f"False Positives: {results['error_analysis']['false_positive_summary']['total_count']}")
    print(f"False Negatives: {results['error_analysis']['false_negative_summary']['total_count']}")
    print("="*80)


if __name__ == "__main__":
    main()