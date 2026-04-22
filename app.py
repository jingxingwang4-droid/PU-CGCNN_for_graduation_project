import streamlit as st
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve

# ================= 页面配置 =================
st.set_page_config(page_title="PU-CGCNN Model Evaluation", page_icon="🔬", layout="wide")

# ================= 侧边栏配置 =================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50)
st.sidebar.title("控制面板")
work_dir = st.sidebar.text_input("工作目录", r"E:\pu-cgcnn")
truth_file = st.sidebar.text_input("测试集标签文件", r"E:\pu-cgcnn\test_dataset\id_prop.csv")
threshold = st.sidebar.slider("可合成概率阈值 (CLscore)", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("**Hardware:** RTX 5070 Ti Laptop\n\n**Framework:** PyTorch & CGCNN")

# ================= 数据加载缓存函数 =================
@st.cache_data
def load_and_evaluate(work_dir, truth_file, threshold):
    # 读取真实标签
    try:
        truth_df = pd.read_csv(truth_file, header=None, names=['material_id', 'true_label'])
        truth_df['material_id'] = truth_df['material_id'].astype(str).str.strip()
        truth_dict = dict(zip(truth_df['material_id'], truth_df['true_label']))
    except Exception as e:
        return None, f"读取标签失败: {e}"

    # 搜集预测结果
    bag_files = glob.glob(os.path.join(work_dir, "test_results_prediction_*.csv"))
    if not bag_files:
        bag_files = glob.glob(os.path.join(work_dir, "test_results_bag_*.csv"))
    if not bag_files:
        return None, "未找到预测结果文件(test_results_*.csv)。"

    pred_dict = defaultdict(list)
    for file in bag_files:
        df = pd.read_csv(file, header=None, names=['material_id', 'target', 'probability'])
        df['material_id'] = df['material_id'].astype(str).str.strip()
        for _, row in df.iterrows():
            pred_dict[row['material_id']].append(row['probability'])

    results = []
    y_true_all, y_scores_all, y_pred_all = [], [], []
    
    for mat_id, probs in pred_dict.items():
        cl_score = float(np.mean(probs))
        is_synthesizable = 1 if cl_score > threshold else 0
        
        if mat_id in truth_dict:
            true_label = int(truth_dict[mat_id])
            y_true_all.append(true_label)
            y_scores_all.append(cl_score)
            y_pred_all.append(is_synthesizable)
        else:
            true_label = -1 # Unknown
            
        results.append({
            'Material_ID': mat_id,
            'True_Label': true_label,
            'CLscore': cl_score,
            'Predicted': is_synthesizable,
            'Model_Count': len(probs)
        })

    results_df = pd.DataFrame(results)
    return {
        "df": results_df,
        "y_true": y_true_all,
        "y_scores": y_scores_all,
        "y_pred": y_pred_all,
        "num_models": len(bag_files)
    }, "Success"

# ================= 主页面 UI =================
st.title("🔬 PU-CGCNN: Structure-Based Synthesizability Prediction")
st.markdown("基于局部监督学习 (PU Learning) 与晶体图卷积神经网络 (CGCNN) 的新材料发现评估系统。")

# 创建三大 Tab 标签页
tab1, tab2, tab3 = st.tabs(["🏛️ 模型架构与总览", "📊 评估指标与混淆矩阵", "💎 AI 新材料发现"])

# --- Tab 1: 架构与总览 ---
with tab1:
    st.header("1. 算法架构与集成策略 (Architecture & Ensemble Strategy)")
    
    # 顶部：宏观的架构图
    img_path = os.path.join(work_dir, "Academic_Architecture_Diagram.png")
    if os.path.exists(img_path):
        st.image(img_path, caption="PU-CGCNN Ensemble Architecture", use_container_width=True)
    else:
        st.warning(f"架构图未找到，请确保已运行之前的可视化脚本并生成在 {work_dir} 下。")
    
    st.markdown("---")
    st.subheader("⚙️ 基学习器深度解析：CGCNN (Crystal Graph Convolutional Neural Network)")
    st.markdown("在本框架中，集成学习的每一个基学习器都是一个独立的 CGCNN 模型。它们负责将 3D 晶体结构映射为高维特征，并输出单模型的合成概率。其核心工作流包含以下三个阶段：")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        with st.expander("📌 阶段一：晶体图构建 (Graph Construction)", expanded=True):
            st.markdown("""
            将 CIF 文件中的周期性晶体转化为图数据结构 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$：
            * **节点 (Nodes, $\mathcal{V}$)**：代表原子。初始特征 $\mathbf{v}_i^{(0)}$ 为 64 维的 One-hot 编码（包含元素周期表属性，如电负性、半径等）。
            * **边 (Edges, $\mathcal{E}$)**：代表原子间的化学键。特征 $\mathbf{u}_{i,j}$ 基于原子间距离进行高斯展开（Gaussian Expansion），维度为 41。
            """)
            
    with col_b:
        with st.expander("🌀 阶段二：图卷积消息传递 (Message Passing)", expanded=True):
            st.markdown("""
            网络的核心。原子节点通过与其相邻原子的相互作用来更新自身的特征表示（通常堆叠 4 层卷积）：
            """)
            st.latex(r"\mathbf{z}_{i,j}^{(l)} = \mathbf{v}_i^{(l)} \oplus \mathbf{v}_j^{(l)} \oplus \mathbf{u}_{i,j}")
            st.markdown("使用 Sigmoid 门控 $\sigma$ 和 Softplus 激活 $g$ 进行特征更新：")
            st.latex(r"\mathbf{v}_i^{(l+1)} = \mathbf{v}_i^{(l)} + \sum_{j} \sigma(\mathbf{z}_{i,j}^{(l)}\mathbf{W}_f) \odot g(\mathbf{z}_{i,j}^{(l)}\mathbf{W}_s)")
            
    with col_c:
        with st.expander("🎯 阶段三：池化与分类 (Pooling & Readout)", expanded=True):
            st.markdown("""
            * **全局池化 (Normalized Softplus)**：将所有局部原子特征聚合成一个固定维度的全局晶体特征向量 $\mathbf{v}_c$。
            * **多层感知机 (MLP)**：特征向量经过全连接层（$64 \rightarrow 128 \rightarrow 2$）。
            * **输出**：通过 LogSoftmax 输出该模型对当前晶体是否可合成的预测概率 $\hat{y}^{(t)}$。
            """)

    st.markdown("---")
    st.subheader("🧠 为什么需要 50 个基学习器？(Bagging 的数学奥义)")
    st.info("""
    **局部监督学习 (PU Learning) 的核心挑战在于我们只有“已合成(1)”的数据，而剩下的(0)中混杂了大量真实的负样本和隐藏的正样本。**
    
    如果只训练一个模型，它极易对抽样到的假负样本产生过拟合。通过 **Bootstrap Aggregating (Bagging)**，我们训练了 50 个 CGCNN：
    1. **强制平衡**：每个模型在训练时，强制抽取与正样本等量的未知数据作为负样本。
    2. **引入多样性**：由于每次抽取的负样本子集不同，50 个模型各自学到了不同的决策边界。
    3. **群体智慧**：真正的不可合成材料会被绝大多数模型打低分，而隐藏的“宝藏材料”则能通过多次求均值脱颖而出。
    """)
    st.latex(r"\text{最终打分 } CLscore = \frac{1}{50} \sum_{t=1}^{50} P(y=1 | \mathcal{G}, \theta^{(t)})")

# 加载数据
data_dict, msg = load_and_evaluate(work_dir, truth_file, threshold)

if data_dict is None:
    st.error(msg)
else:
    df = data_dict['df']
    y_true = np.array(data_dict['y_true'])
    y_scores = np.array(data_dict['y_scores'])
    y_pred = np.array(data_dict['y_pred'])
    
    # --- Tab 2: 评估指标 ---
    with tab2:
        st.header(f"2. 测试集评估报告 (基于 {data_dict['num_models']} 个集成模型)")
        
        # 顶部指标卡片
        col1, col2, col3, col4 = st.columns(4)
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_scores)
        else:
            auc = np.nan
            
        col1.metric(label="ROC-AUC Score", value=f"{auc:.4f}" if not np.isnan(auc) else "N/A")
        col2.metric(label="Total Test Samples", value=len(y_true))
        col3.metric(label="Bagging Models", value=data_dict['num_models'])
        col4.metric(label="Decision Threshold", value=threshold)

        st.markdown("---")
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            st.subheader("混淆矩阵 (Confusion Matrix)")
            cm = confusion_matrix(y_true, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Negative (0)', 'Synthesizable (1)'],
                        yticklabels=['Negative (0)', 'Synthesizable (1)'], ax=ax_cm)
            ax_cm.set_xlabel('Predicted Label')
            ax_cm.set_ylabel('True Label')
            st.pyplot(fig_cm)
            
        with col_plot2:
            st.subheader("ROC 曲线 (ROC Curve)")
            if not np.isnan(auc):
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.3f})')
                ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)
            else:
                st.info("数据类别单一，无法绘制 ROC 曲线。")
                
        st.subheader("详细分类报告 (Classification Report)")
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report_dict).transpose(), use_container_width=True)

    # --- Tab 3: 材料发现 ---
    with tab3:
        st.header("3. 高潜力未知材料挖掘 (Discovery of Synthesizable Crystals)")
        st.markdown(f"以下是从标签为 `0` (未知/虚拟) 的数据中，筛选出 `CLscore > {threshold}` 的高分材料。这些是极具实验价值的候选者！")
        
        # 过滤发现的材料并排序
        discoveries = df[(df['True_Label'] == 0) & (df['CLscore'] > threshold)]
        discoveries = discoveries.sort_values(by='CLscore', ascending=False).reset_index(drop=True)
        
        st.success(f"🎉 在测试集中成功挖掘出 **{len(discoveries)}** 个潜在的可合成材料！")
        
        # 交互式数据表展示
        st.dataframe(discoveries[['Material_ID', 'CLscore', 'Model_Count']].style.highlight_max(axis=0, subset=['CLscore']), 
                     use_container_width=True)
        
        st.markdown("### 分数分布直方图 (CLscore Distribution)")
        fig_dist, ax_dist = plt.subplots(figsize=(10, 3))
        sns.histplot(df, x='CLscore', hue='True_Label', bins=50, kde=True, palette='Set2', ax=ax_dist)
        ax_dist.axvline(threshold, color='red', linestyle='--', label='Decision Boundary')
        ax_dist.legend()
        st.pyplot(fig_dist)