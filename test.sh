#!/bin/bash

# YAML 파일 경로 리스트
yaml_files=(
    # "configs/text_guided_split/alien_1.yaml"
    # "configs/text_guided_split/alien_2.yaml"
    # "configs/text_guided_split/alien_3.yaml"
    # "configs/text_guided_split/alien_4.yaml"
    # "configs/text_guided_split/alien_5.yaml"
    # "configs/text_guided_split/rabbit_1.yaml"
    "configs/text_guided_split/rabbit_2.yaml"
    # "configs/text_guided_split/rabbit_3.yaml"
    # "configs/text_guided_split/rabbit_4.yaml"
    # "configs/text_guided_split/rabbit_5.yaml" 
    # "configs/text_guided_split/cat_1.yaml"
    # "configs/text_guided_split/cat_2.yaml"
    # "configs/text_guided_split/cat_3.yaml"
    # "configs/text_guided_split/cat_4.yaml"
    # "configs/text_guided_split/cat_5.yaml"
)

# 로그 파일 저장 경로
mkdir -p logs
log_file="logs/log_T29.txt"
echo "Execution Log" > $log_file
echo "==========================" >> $log_file
echo "=========================================" >> $log_file
echo "|         Parameter Change Log          |" >> $log_file
echo "=========================================" >> $log_file
echo "| Parameter         | Default | Set     |" >> $log_file
echo "|-------------------|---------|---------|" >> $log_file
echo "| Gaussian Blur K  | 21      | 9       |" >> $log_file
echo "| Gaussian Blur S  | 16      | 3       |" >> $log_file
echo "| Erode Kernel     | 5       | 11      |" >> $log_file
echo "| Dilate Kernel    | 25      | 9       |" >> $log_file
echo "| z_update_thr     | 0.2     | -0.1    |" >> $log_file
echo "| n_views          | 8       | 8       |" >> $log_file
echo "=========================================" >> $log_file
echo "| Grid batch order set                  |" >> $log_file
echo "=========================================" >> $log_file



echo "==========================" >> $log_file
# YAML 파일별 실행
for yaml in "${yaml_files[@]}"; do
    echo "Starting: $yaml" | tee -a $log_file
    start_time=$(date +%s)

    # VRAM 사용량 측정 (Python 실행 전)
    vram_before=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR==1')

    # Python 실행 (CUDA_VISIBLE_DEVICES=1 설정)
    CUDA_VISIBLE_DEVICES=1 python -m scripts.run_texture --config_path="$yaml"

    # VRAM 사용량 측정 (Python 실행 후)
    vram_after=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR==1')

    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))

    # 결과 기록
    echo "Finished: $yaml" | tee -a $log_file
    echo "Elapsed Time: $elapsed_time seconds" | tee -a $log_file
    echo "Max VRAM Used: $vram_max MB" | tee -a $log_file
    echo "--------------------------" >> $log_file
done
