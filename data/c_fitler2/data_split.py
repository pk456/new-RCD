import json

with open('log_data.json', 'r') as f:
    data = json.load(f)

# 选出其中80%的数据
train_data = data[:int(len(data) * 0.8)]
test_data = data[int(len(data) * 0.8):]

with open(f'train_set.json', 'w', encoding='utf8') as output_file:
    json.dump(train_data, output_file, indent=4, ensure_ascii=False)
with open(f'test_set.json', 'w', encoding='utf8') as output_file:
    json.dump(test_data, output_file, indent=4, ensure_ascii=False)  # 直接用test_set作为val_set
