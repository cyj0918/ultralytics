# yolo train model=yolo11n.pt data=ultralytics/cfg/datasets/mf.yaml epochs=10 batch=16 device=mps
# yolo train model=ultralytics/cfg/models/modified/yolo11-1.yaml data=ultralytics/cfg/datasets/mf.yaml epochs=100 imgsz=224 batch=4 device=mps

# yolo predict model=runs/detect/train4/weights/best.pt source=/Users/jhen/Documents/CUHK-Project/dataset/yolo_worker_training/images/test/
# yolo detect val data=yolo11n.pt data=ultralytics/cfg/datasets/mf.yaml model=runs\detect\train4\weights\best.pt batch=16 device=0

# Train: yolo detect train data=ultralytics/cfg/datasets/mf.yaml model=ultralytics/cfg/models/modified/exp1.yaml epochs=100 batch=16 lr0=0.01 optimizer=SGD device=mps plots=True save=True project=exp1 name=exp1
# Detect: python3 tests/exp_evaluate.py --source /Users/jhen/Documents/CUHK-Project/dataset/yolo_worker_training/images/test/  --dataset-config ultralytics/cfg/datasets/mf.yaml --model /Users/jhen/Documents/CUHK-Project/ultralytics/exp0/exp0/weights/best.pt --project exp0 --name test0-1
# Statistic: python3 tests/exp_statistic.py --gt-dir /Users/jhen/Documents/CUHK-Project/dataset/yolo_worker_training/labels/test/ --pred-dir /Users/jhen/Documents/CUHK-Project/ultralytics/exp0/test0-3/labels --output /Users/jhen/Documents/CUHK-Project/ultralytics/exp0/test0-3/statistics.json