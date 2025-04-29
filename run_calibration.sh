#!/bin/bash

python run_calibration.py \
  --img_root data/Image \
  --roi_pcd_root data/RoIPCD \
  --whole_pcd_root data/WholePCD \
  --calibration_target_path configs/camera_calibration_parameters_velodyne.yaml \
  --num_row 6 \
  --num_col 5 \
  --square 150 \
  # --transform_matrix_path results/28-04-2025-20-44-40