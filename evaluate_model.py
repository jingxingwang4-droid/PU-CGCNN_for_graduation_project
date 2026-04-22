import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict
try:
    from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
except ImportError:
    print("[!] 缺少 sklearn 库，请先运行: pip install scikit-learn")
    exit()

def main():
    # ================= 配置区 =================
    work_dir = r"E:\pu-cgcnn"
    ground_truth_file = r"E:\pu-cgcnn\test_dataset\id_prop.csv"
    threshold = 0.5  # 论文设定的合成概率阈值 CLscore > 0.5
    # ==========================================

    print("="*65)
    print("🚀 PU-CGCNN 综合模型评估与发现程序 (Ensemble Evaluation)")
    print("="*65)

    # 1. 读取真实标签 (Ground Truth)
    print(f"[*] 正在加载真实标签: {ground_truth_file}...")
    try:
        truth_df = pd.read_csv(ground_truth_file, header=None, names=['material_id', 'true_label'])
        # 【关键修复1】强制转为字符串并去除两端空格，防止格式对不上
        truth_df['material_id'] = truth_df['material_id'].astype(str).str.strip()
        truth_dict = dict(zip(truth_df['material_id'], truth_df['true_label']))
    except Exception as e:
        print(f"[!] 致命错误: 读取真实标签失败。详细信息: {e}")
        return

    # 2. 搜集并融合所有的预测结果
    # 【关键修复2】兼容 predict 脚本和 main 脚本生成的不同文件名
    bag_files = glob.glob(os.path.join(work_dir, "test_results_prediction_*.csv"))
    if not bag_files:
        bag_files = glob.glob(os.path.join(work_dir, "test_results_bag_*.csv"))
    
    if not bag_files:
        print(f"[!] 错误: 在 {work_dir} 下未找到任何预测结果文件(test_results_*.csv)！")
        return
        
    print(f"[*] 成功探测到 {len(bag_files)} 个独立模型的预测结果，正在进行概率融合(Ensemble)...")
    
    pred_dict = defaultdict(list)
    for file in bag_files:
        df = pd.read_csv(file, header=None, names=['material_id', 'target', 'probability'])
        # 清理 ID 空格
        df['material_id'] = df['material_id'].astype(str).str.strip()
        for _, row in df.iterrows():
            pred_dict[row['material_id']].append(row['probability'])
            
    # 3. 数据对齐与核心计算
    results = []
    y_true_all = []
    y_scores_all = []
    y_pred_all = []
    
    match_count = 0

    for mat_id, probs in pred_dict.items():
        # 计算 CLscore (平均预测概率)
        cl_score = float(np.mean(probs))
        
        # 判断是否可合成
        is_synthesizable = 1 if cl_score > threshold else 0
        
        # 尝试匹配真实标签
        if mat_id in truth_dict:
            true_label = int(truth_dict[mat_id])
            y_true_all.append(true_label)
            y_scores_all.append(cl_score)
            y_pred_all.append(is_synthesizable)
            match_count += 1
        else:
            true_label = "Unknown"
            
        results.append({
            'Material_ID': mat_id,
            'True_Label': true_label,
            'CLscore': cl_score,
            'Predicted': "Yes" if is_synthesizable else "No",
            'Model_Count': len(probs)
        })

    # 【关键修复3】拦截空样本错误，并给出明确排查提示
    if match_count == 0:
        print("\n[!] 严重错误：预测结果中的物质 ID 与 id_prop.csv 中的物质 ID 没有任何重合！")
        print("💡 请检查：你是否用训练集的 csv 去匹配了测试集的预测结果？")
        print(f"-> 预测结果里的 ID 示例: {list(pred_dict.keys())[:3]}")
        print(f"-> 真实标签里的 ID 示例: {list(truth_dict.keys())[:3]}")
        return
        
    print(f"[*] 成功对齐 {match_count} 个具有真实标签的材料数据，正在计算科学指标...")

    # 4. 科学计算评估指标
    # 【关键修复4】防止测试集中全为 0 导致 AUC 报错
    unique_labels = np.unique(y_true_all)
    if len(unique_labels) > 1:
        auc_score = roc_auc_score(y_true_all, y_scores_all)
    else:
        auc_score = float('nan')
        
    cm = confusion_matrix(y_true_all, y_pred_all)
    report = classification_report(y_true_all, y_pred_all, zero_division=0)

    # 5. 打印最终论文级报告
    print("\n" + "="*65)
    print("📊 PU-CGCNN 测试集评估报告 (Test Set Report)")
    print("="*65)
    
    if not np.isnan(auc_score):
        print(f"⭐ 核心指标 ROC-AUC Score : {auc_score:.4f}")
    else:
        print(f"⭐ 核心指标 ROC-AUC Score : 无法计算 (测试集中仅包含类别 {unique_labels})")
        
    print("\n📈 混淆矩阵:")
    print("          [预测不可合成] [预测可合成]")
    print(f"[真实不可合成(0)]  {cm[0][0]:<12} {cm[0][1]}")
    if len(cm) > 1:
        print(f"[真实可合成(1)]    {cm[1][0]:<12} {cm[1][1]}")
    
    print("\n📝 详细分类指标 (Precision / Recall / F1):")
    print(report)
    print("="*65)

    # 6. 保存与挖掘新材料
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='CLscore', ascending=False).reset_index(drop=True)
    
    output_csv = os.path.join(work_dir, "Final_Ensemble_Predictions.csv")
    results_df.to_csv(output_csv, index=False)
    print(f"\n[*] 预测完毕！完整的合成概率打分表已保存至: {output_csv}")
    
    print("\n🏆 【AI 新材料发现】Top 5 最具合成潜力的未知材料:")
    # 过滤出真实标签为 0 (未知/未合成) 并且模型打分极高的材料
    discoveries = results_df[results_df['True_Label'] == 0].head(5)
    for _, row in discoveries.iterrows():
        print(f"   - 晶体 ID: {row['Material_ID']:<12} | 合成概率 (CLscore) = {row['CLscore']:.4f}")

if __name__ == "__main__":
    main()