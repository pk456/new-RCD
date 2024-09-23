import json

with open('../data/c/log_data.json', encoding='utf-8') as f:
    data = json.load(f)
print(len(data))
for item in data:
    for log in item['logs']:
        if len(log['knowledge_code']) >= 2:
            print(item)
            break
