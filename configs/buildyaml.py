import os
import yaml


class CustomDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(CustomDumper, self).increase_indent(flow, False)

# YAML 파일 작성 함수
def dump_yaml(data, file_path):
    with open(file_path, "w") as yaml_file:
        yaml.dump(data, yaml_file, Dumper=CustomDumper, default_flow_style=False, sort_keys=False)


# 기존 YAML 파일들이 위치한 디렉토리와 분할된 파일 저장 디렉토리
source_dir = "./text_guided"
output_dir = "./text_guided_split"
os.makedirs(output_dir, exist_ok=True)

# 모든 YAML 파일 읽기
yaml_files = [f for f in os.listdir(source_dir) if f.endswith(".yaml")]

for yaml_file in yaml_files:
    source_path = os.path.join(source_dir, yaml_file)
    
    # 원본 YAML 파일 읽기
    with open(source_path, "r") as file:
        data = yaml.safe_load(file)
    
    # prompt 리스트를 분리
    if "guide" in data and "text" in data["guide"] and isinstance(data["guide"]["text"], list):
        prompts = data["guide"]["text"]
    else:
        print(f"Skipping file (no prompt list found): {yaml_file}")
        continue

    # 각 프롬프트별로 새로운 YAML 파일 생성
    base_name = os.path.splitext(yaml_file)[0]  # 파일 이름에서 확장자 제거
    for idx, prompt in enumerate(prompts, start=1):
        # 원본 데이터를 복사하여 수정
        new_data = data.copy()
        new_data["log"]["exp_name"] = f"{base_name}_{idx}"
        new_data["guide"]["text"] = f"{prompt}"  # 이중 따옴표가 그대로 유지되도록 설정

        # 새로운 파일 경로
        new_file_name = f"{base_name}_{idx}.yaml"
        new_file_path = os.path.join(output_dir, new_file_name)

        # YAML 파일 저장
        dump_yaml(new_data, new_file_path)
        print(f"Generated: {new_file_path}")
