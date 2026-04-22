import os
import sys
import subprocess
import glob
import pickle
import pandas as pd
import numpy as np
import shutil
from collections import defaultdict

def main():
    # ================= 配置区 =================
    work_dir = r"E:\pu-cgcnn"
    target_cifs_dir = os.path.join(work_dir, "target_data")
    target_graph_dir = os.path.join(work_dir, "target_graph")
    model_dir = work_dir  
    bag_num = 50          
    
    # 动态将 src_code 注入环境变量，保证我们的脚本能随时调用底层算法
    src_dir = os.path.join(work_dir, "src_code")
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    # ==========================================

    print("="*65)
    print("🚀 PU-CGCNN 新材料打分管线 (单线程安全直驱版)")
    print("="*65)

    # 0. 防呆设计：同步 atom_init.json
    atom_src = os.path.join(target_cifs_dir, "atom_init.json")
    atom_dst = os.path.join(work_dir, "atom_init.json")
    if os.path.exists(atom_src) and not os.path.exists(atom_dst):
        shutil.copy(atom_src, atom_dst)

    # 1. 扫描真实文件，强制重写 id_prop.csv
    cif_files = glob.glob(os.path.join(target_cifs_dir, "*.cif"))
    if not cif_files:
        print(f"[!] 严重错误：在 {target_cifs_dir} 下找不到任何 .cif 文件！")
        return
        
    print(f"[*] 探测到 {len(cif_files)} 个 CIF 文件，正在重写 id_prop.csv...")
    valid_lines = [f"{os.path.basename(f)[:-4]},0\n" for f in cif_files]
    id_prop_path = os.path.join(target_cifs_dir, "id_prop.csv")
    with open(id_prop_path, "w", encoding="utf-8") as f:
        f.writelines(valid_lines)

    # 2. 清理历史文件
    for f in glob.glob(os.path.join(work_dir, "test_results_prediction_*.csv")) + \
             glob.glob(os.path.join(work_dir, "test_results_bag_*.csv")):
        try: os.remove(f)
        except: pass
        
    if not os.path.exists(target_graph_dir):
        os.makedirs(target_graph_dir)
    else:
        for f in glob.glob(os.path.join(target_graph_dir, "*.pickle")):
            os.remove(f)

    # ================= 核心修复：绕过多进程 Bug，单线程亲手构图 =================
    print(f"\n[*] 第一步：正在安全模式下构建图网络 (绕过 Windows 多进程 Bug)...")
    try:
        from cgcnn.data_PU_learning import CIFData
        dataset = CIFData(target_cifs_dir)
        
        # 实时打印进度
        for i in range(len(dataset)):
            data_tuple = dataset[i]
            cif_id = dataset.id_prop_data[i][0]
            save_path = os.path.join(target_graph_dir, f"{cif_id}.pickle")
            
            with open(save_path, 'wb') as f:
                pickle.dump(data_tuple, f)
            
            # 每处理 10 个打印一次进度，让你心里有底
            if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
                print(f"    -> 已成功构建: {i + 1}/{len(dataset)} 个")
                
    except Exception as e:
        print(f"\n[!] 构图过程发生错误: {e}")
        return

    generated_pickles = glob.glob(os.path.join(target_graph_dir, "*.pickle"))
    if len(generated_pickles) == 0:
        print("\n[!] 图网络文件仍未生成，请检查报错。")
        return
    else:
        print(f"[*] 完美！成功将 {len(generated_pickles)} 个晶体转化为图网络文件。")

    # 4. 调度模型流水线打分
    print(f"\n[*] 第二步：正在调用 {bag_num} 个底层模型进行合成概率预测...")
    cmd_predict = f"python src_code/predict_PU_learning.py --bag {bag_num} --graph {target_graph_dir} --cifs {target_cifs_dir} --modeldir {model_dir}"
    try:
        subprocess.run(cmd_predict, shell=True, check=True)
    except subprocess.CalledProcessError:
        print("[!] 预测脚本运行失败，请检查命令参数。")
        return

    # 5. 融合打分结果并排序
    print(f"\n[*] 第三步：正在融合模型打分，生成最终降序报告...")
    bag_files = glob.glob(os.path.join(work_dir, "test_results_prediction_*.csv"))
    if not bag_files:
        bag_files = glob.glob(os.path.join(work_dir, "test_results_bag_*.csv"))
        
    pred_dict = defaultdict(list)
    for file in bag_files:
        df = pd.read_csv(file, header=None, names=['material_id', 'target', 'probability'])
        df['material_id'] = df['material_id'].astype(str).str.strip()
        for _, row in df.iterrows():
            pred_dict[row['material_id']].append(row['probability'])
            
    results = [{'Material_ID': mat_id, 'CLscore': float(np.mean(probs))} for mat_id, probs in pred_dict.items()]
    results_df = pd.DataFrame(results).sort_values(by='CLscore', ascending=False).reset_index(drop=True)
    
    # 输出结果
    output_csv = os.path.join(work_dir, "Target_Data_CLscore_Rank.csv")
    results_df.to_csv(output_csv, index=False)
    
    print("\n" + "="*65)
    print("🏆 预测完成！最具合成潜力的候选材料 Top 10：")
    print("-" * 65)
    print(f"{'Material ID':<20} | {'CLscore (合成概率)':<15}")
    print("-" * 65)
    for _, row in results_df.head(10).iterrows():
        star = "⭐ (高潜力)" if row['CLscore'] > 0.5 else ""
        print(f"{row['Material_ID']:<20} | {row['CLscore']:.6f}  {star}")
    print("="*65)
    print(f"[*] 完整的 68 个材料打分排名已保存至: {output_csv}")

if __name__ == "__main__":
    main()