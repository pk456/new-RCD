import json

with open(f'../data/ASSIST/log_data_all.json', encoding='utf8') as i_f:
    stus = json.load(i_f)

exer = set()
kn = set()

for stu in stus:
    for log in stu['logs']:
        exer.add(log['exer_id'])
        for k in log['knowledge_code']:
            kn.add(k)
print(len(exer), len(kn)) # 413 413
