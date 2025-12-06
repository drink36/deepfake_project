import json
with open('validation_metadata_filtered.json', 'r') as f:
    data= json.load(f)
# path file and ground truth file
with open('output.txt', 'w') as f:
    for item in data:
        if(item['split'] == 'train'):
        # 寫入路徑並換行
            f.write(item['file'] + '\n')
with open('output.txt', 'w') as f:
    for item in data:
        if(item['split'] == 'train'):
        # 寫入路徑並換行
            f.write(item['modify_type'] + '\n')
