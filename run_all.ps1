Write-Host "1. Starting Ablation Study..."
python "d:/Fed learning project/scripts/run_ablation_study.py"

Write-Host "2. Starting YOLOv8 Baseline..."
python "d:/Fed learning project/scripts/train_yolov8_baseline.py"

Write-Host "3. Starting Faster R-CNN Baseline..."
python "d:/Fed learning project/scripts/train_faster_rcnn_baseline.py"

Write-Host "4. Starting Statistical Validation..."
python "d:/Fed learning project/scripts/run_statistical_validation.py"

Write-Host "ALL TRAINING COMPLETE!"
