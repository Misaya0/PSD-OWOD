import json
import os

# 假设你的JSON文件都存放在这个目录下
directory = '/home/hebut-02/common/gzj/OrthogonalDet-main/datasets/SAM_H'

# 遍历目录中的所有文件
for filename in os.listdir(directory):
    if filename.endswith('.json'):  # 确保是JSON文件
        file_path = os.path.join(directory, filename)

        try:
            with open(file_path, 'r') as file:
                data = json.load(file)

                # 检查'box_result'键是否存在且不为空
                if 'box_result' in data and data['box_result']:
                    # 进一步检查'box_result'列表中的每个元素是否为空
                    if any(item.get('bbox') is None or item.get('score') is None for item in data['box_result']):
                        print(f'File {filename} has empty values.')
                    else:
                        print(f'File {filename}'+'is valid.')
                else:
                    print(f'File {filename} is empty or missing "box_result" key.')
        except json.JSONDecodeError:
            print(f'File {filename} is not a valid JSON file.')
        except Exception as e:
            print(f'An error occurred with file {filename}: {e}')