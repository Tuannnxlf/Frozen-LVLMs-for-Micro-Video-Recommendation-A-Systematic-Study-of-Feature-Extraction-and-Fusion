import csv

filepath = '/opt/data/private/work/vllmembs4rec/test.csv'

result = [[1, 2],[3, 4]]

with open(filepath, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # 写入表头
    writer.writerow(['UserID', 'Rank'])
    # 写入数据
    writer.writerows(result)