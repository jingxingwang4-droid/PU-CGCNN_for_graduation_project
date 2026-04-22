import os
import shutil
import csv

# 定义你的路径
test_csv = r"E:\pu-cgcnn\test_dataset\id_prop.csv"
src_dir = r"E:\pu-cgcnn\saved_crystal_graph"
dest_dir = r"E:\pu-cgcnn\saved_test_graph"

# 自动创建目标文件夹
os.makedirs(dest_dir, exist_ok=True)

count = 0
print("正在从十万大军中精准提取测试集数据，请稍候...")

# 读取测试集 ID 并进行移动
with open(test_csv, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row: continue
        cif_id = row[0]
        file_name = cif_id + ".pickle"
        src_file = os.path.join(src_dir, file_name)
        dest_file = os.path.join(dest_dir, file_name)
        
        # 找到文件就移动
        if os.path.exists(src_file):
            shutil.move(src_file, dest_file)
            count += 1

print(f"✅ 大功告成！成功分离并移动了 {count} 个测试集图文件到 saved_test_graph 文件夹！")