# yolo train model=yolo11n.pt data=ultralytics/cfg/datasets/mf.yaml epochs=10 batch=16 device=mps
# yolo train model=ultralytics/cfg/models/modified/yolo11-1.yaml data=ultralytics/cfg/datasets/mf.yaml epochs=100 imgsz=224 batch=4 device=mps

# yolo predict model=runs/detect/train4/weights/best.pt source=/Users/jhen/Documents/CUHK-Project/dataset/yolo_worker_training/images/test/
# yolo detect val data=yolo11n.pt data=ultralytics/cfg/datasets/mf.yaml model=runs\detect\train4\weights\best.pt batch=16 device=0
