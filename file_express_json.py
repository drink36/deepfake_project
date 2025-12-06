import json

# Read original JSON
with open('validation_metadata_filtered.json', 'r') as f:
    data = json.load(f)
train_data = []
for item in data:
    if item.get('split') == 'train': # 使用 .get 防止 key 不存在報錯
        train_data.append(item)
# Take first 3 entries
first_three = train_data[:300]

# Save to new JSON file
with open('first_test1.json', 'w') as f:
    print('do')
    json.dump(first_three, f, indent=2)
