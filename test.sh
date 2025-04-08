#!/bin/bash

# YAML 파일 경로 리스트
yaml_files=(
    # "configs/text_guided_split/rabbit_6.yaml"
    "configs/text_guided_split/alien_1.yaml"
    "configs/text_guided_split/alien_2.yaml"
    "configs/text_guided_split/alien_3.yaml"
    "configs/text_guided_split/alien_4.yaml"
    "configs/text_guided_split/alien_5.yaml"
    "configs/text_guided_split/napoleon_1.yaml"
    "configs/text_guided_split/napoleon_2.yaml"
    "configs/text_guided_split/napoleon_3.yaml"
    "configs/text_guided_split/napoleon_4.yaml"
    "configs/text_guided_split/napoleon_5.yaml"
    "configs/text_guided_split/nascar_1.yaml"
    "configs/text_guided_split/nascar_2.yaml"
    "configs/text_guided_split/nascar_3.yaml"
    "configs/text_guided_split/nascar_4.yaml"
    "configs/text_guided_split/nascar_5.yaml"
    "configs/text_guided_split/rabbit_1.yaml"
    "configs/text_guided_split/rabbit_2.yaml"
    # "configs/text_guided_split/rabbit_7.yaml"
    "configs/text_guided_split/rabbit_3.yaml"
    "configs/text_guided_split/rabbit_4.yaml"
    "configs/text_guided_split/rabbit_5.yaml"
)

# 로그 파일 저장 경로
mkdir -p logs
log_file="logs/log_T61.txt"
echo "Execution Log" > $log_file
echo "==========================" >> $log_file
echo "=========================================" >> $log_file
echo "|         Parameter Change Log         |" >> $log_file
echo "=========================================" >> $log_file
echo "| Parameter        | Default | Set     |" >> $log_file
echo "|------------------|---------|---------|" >> $log_file
echo "| Gaussian Blur K  | 21      | 21      |" >> $log_file
echo "| Gaussian Blur S  | 16      | 16      |" >> $log_file
echo "| Erode Kernel     | 5       | 5       |" >> $log_file
echo "| Dilate Kernel    | 25      | 25      |" >> $log_file
echo "| z_update_thr     | 0.2     | -0.1    |" >> $log_file
echo "| n_views          | 8       | 4       |" >> $log_file
echo "| ControlNet Scale | 1.0     | 0.5     |" >> $log_file
echo "| Inference Step   | 50      | 50      |" >> $log_file
echo "| Optimize  Step   | 200     | 200     |" >> $log_file
echo "=========================================" >> $log_file
echo "| No checker Mask to Refine            |" >> $log_file
echo "| Generate Front at the last Phase     |" >> $log_file
# echo "| No BLD at the first Stage            |" >> $log_file
echo "| No Phi Sampling, CW Rotation         |" >> $log_file
echo "| Add Negative Prompts                 |" >> $log_file
echo "| refine_mask[z_normals < 0.6 ] = 0    |" >> $log_file
echo "=========================================" >> $log_file



echo "==========================" >> $log_file
# YAML 파일별 실행
for yaml in "${yaml_files[@]}"; do
    echo "Starting: $yaml" | tee -a $log_file
    start_time=$(date +%s)

    # Python 실행 (CUDA_VISIBLE_DEVICES=1 설정)
    CUDA_VISIBLE_DEVICES=1 python -m scripts.run_texture --config_path="$yaml"

    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))

    # 결과 기록
    echo "Finished: $yaml" | tee -a $log_file
    echo "Elapsed Time: $elapsed_time seconds" | tee -a $log_file
    echo "--------------------------" >> $log_file
done
