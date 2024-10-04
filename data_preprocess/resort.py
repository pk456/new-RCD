import json

min_records = 15
with open('../data/c_fitler2/log_data_old.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print("before filter: ", len(data))
# result = [student_record for student_record in data if student_record['log_num'] > min_records]
# print("after filter: ", len(result))

new_stu_map = {}
new_stu_num = 0
new_exer_map = {}
new_exer_num = 0
for log in data:
    # 如果student_map的key中没有user_id，
    stu_key = log['user_id']
    if stu_key not in new_stu_map:
        new_stu_map[stu_key] = new_stu_num
        new_stu_num += 1
    log['user_id'] = new_stu_map[stu_key]
    exer_key = log['exer_id']
    if exer_key not in new_exer_map:
        new_exer_map[exer_key] = new_exer_num
        new_exer_num += 1
    log['exer_id'] = new_exer_map[exer_key]
with open('../data/c_fitler2/log_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
with open('../data/c_fitler2/exer_map.json', 'w', encoding='utf-8') as f:
    json.dump(new_exer_map, f, ensure_ascii=False, indent=4)
print("exer number: ",len(new_exer_map))
with open('../data/c_fitler2/stu_map.json', 'w', encoding='utf-8') as f:
    json.dump(new_stu_map, f, ensure_ascii=False, indent=4)
print("stu number: ",len(new_stu_map))
