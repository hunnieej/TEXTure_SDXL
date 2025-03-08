#!/bin/bash

# YAML 파일 경로 리스트
yaml_files=(
    "configs/text_guided_split/alien_1.yaml"
    "configs/text_guided_split/alien_2.yaml"
    "configs/text_guided_split/alien_3.yaml"
    "configs/text_guided_split/alien_4.yaml"
    "configs/text_guided_split/alien_5.yaml"
    "configs/text_guided_split/rabbit_1.yaml"
    "configs/text_guided_split/rabbit_2.yaml"
    "configs/text_guided_split/rabbit_3.yaml"
    "configs/text_guided_split/rabbit_4.yaml"
    "configs/text_guided_split/rabbit_5.yaml" 
    "configs/text_guided_split/sphere_1.yaml"
    "configs/text_guided_split/sphere_2.yaml"
    "configs/text_guided_split/sphere_3.yaml"
    "configs/text_guided_split/sphere_4.yaml"
    "configs/text_guided_split/sphere_5.yaml"
)

# 로그 파일 저장 경로
mkdir -p logs
log_file="logs/log_sdxl_test_24.txt"
echo "Execution Log" > $log_file
echo "==========================" >> $log_file
echo "T18" >> $log_file
echo "Scheduler : DPMMultstepscheduler with kerras_sigmas=True" >> $log_file
echo "theta 60 to 40" >> $log_file

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

    # 최대 VRAM 사용량 계산
    vram_max=$((vram_after - vram_before))

    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))

    # 결과 기록
    echo "Finished: $yaml" | tee -a $log_file
    echo "Elapsed Time: $elapsed_time seconds" | tee -a $log_file
    echo "Max VRAM Used: $vram_max MB" | tee -a $log_file
    echo "--------------------------" >> $log_file
done
