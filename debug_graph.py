import os
import sys
import traceback

work_dir = r"E:\pu-cgcnn"
# 【关键修复】将 src_code 加入环境变量，确保能精准找到 cgcnn 模块
src_dir = os.path.join(work_dir, "src_code")
sys.path.append(src_dir)

target_cifs_dir = os.path.join(work_dir, "target_data")

print("="*60)
print("🔬 深度显微镜 (修复版)：精准定位跳过原因")
print("="*60)

try:
    print("[*] 正在导入底层图网络数据集模块 (CIFData)...")
    from cgcnn.data_PU_learning import CIFData
    
    print("[*] 正在初始化目标数据集...")
    dataset = CIFData(target_cifs_dir)
    print(f"[*] 数据集初始化成功，已读取 id_prop.csv，包含 {len(dataset)} 个目标。")
    
    # 打印一下它准备处理哪个文件，防止是文件名读错了
    target_id = dataset.id_prop_data[0][0]
    print(f"[*] 正在强制对第一个晶体 ({target_id}.cif) 执行图转化计算...")
    
    # 这里是触发底层的核心地带
    data = dataset[0]
    
    print("\n🎉 解析成功！如果看到这条消息，说明底层一切正常。")
    
except Exception as e:
    print("\n[!!!] 抓到静默跳过的元凶了！底层的真实报错如下：\n")
    traceback.print_exc()
    print("\n" + "="*60)
    print("💡 请直接把上面红色的 Traceback 报错信息复制发给我，真相就在里面！")