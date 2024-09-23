import json

min_records = 15
with open('../data/c/log_data_all.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print("before filter: ", len(data))
result = [student_record for student_record in data if student_record['log_num'] > min_records]
print("after filter: ", len(result))

new_stu_map = {}
new_stu_num = 0
new_exer_map = {}
new_exer_num = 0
for stu_record in result:
    # 如果student_map的key中没有user_id，
    stu_key = stu_record['user_id']
    if stu_key not in new_stu_map:
        new_stu_map[stu_key] = new_stu_num
        new_stu_num += 1
    stu_record['user_id'] = new_stu_map[stu_key]
    for log in stu_record['logs']:
        exer_key = log['exer_id']
        if exer_key not in new_exer_map:
            new_exer_map[exer_key] = new_exer_num
            new_exer_num += 1
        log['exer_id'] = new_exer_map[exer_key]
with open('../data/c_filter/log_data.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
with open('../data/c_filter/exer_map.json', 'w', encoding='utf-8') as f:
    json.dump(new_exer_map, f, ensure_ascii=False, indent=4)
with open('../data/c_filter/stu_map.json', 'w', encoding='utf-8') as f:
    json.dump(new_stu_map, f, ensure_ascii=False, indent=4)
