import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_academic_architecture_large(output_path):
    """绘制带有严谨张量维度和数学符号的大字号学术版架构图"""
    # 1. 扩大画布尺寸，给大字体留出空间
    fig, ax = plt.subplots(figsize=(22, 12), facecolor='white')
    ax.axis('off')
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 11)
    
    # 辅助绘图函数 (默认字号调大至 14)
    def draw_box(x, y, w, h, text, bg_color='#f8f9fa', edge_color='#212529', font_size=14, font_weight='normal'):
        box = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=edge_color, facecolor=bg_color)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, color='#212529', fontsize=font_size, 
                ha='center', va='center', weight=font_weight)

    def draw_arrow(x1, y1, x2, y2, label="", offset_y=0.2, font_size=13):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='#495057', shrink=0.02, width=1.5, headwidth=8))
        if label:
            # 箭头上方标注字体加大并加粗
            ax.text((x1+x2)/2, (y1+y2)/2 + offset_y, label, ha='center', va='center', 
                    fontsize=font_size, color='#d9480f', style='italic', weight='bold')

    # ================= 1. PU Learning 数据采样流 (左侧) =================
    ax.text(2.8, 10.2, "1. PU Bagging Strategy", fontsize=18, weight='bold', ha='center', color='#1864ab')
    
    # 放宽 Box 宽度和高度
    draw_box(0.5, 8.5, 4.6, 1.4, r"Database $\mathcal{D}$" + "\n" + r"Positive $\mathcal{P}$, Unlabeled $\mathcal{U}$", bg_color='#e7f5ff')
    draw_arrow(2.8, 8.5, 2.8, 7.3, r"Subset Extraction")
    
    draw_box(0.5, 5.5, 4.6, 1.8, r"Iteration $t \in \{1, \dots, T\}$" + "\n\n" + 
             r"$\mathcal{D}_{train}^{(t)} = \mathcal{P} \cup \mathcal{U}_n^{(t)}$" + "\n" + 
             r"$|\mathcal{P}| = |\mathcal{U}_n^{(t)}| = K$", bg_color='#fff3bf')
    
    draw_arrow(2.8, 5.5, 2.8, 4.3, r"Mini-Batch Sampling")
    draw_box(0.5, 3.0, 4.6, 1.3, r"Input Batch $\mathcal{B}$" + "\n" + r"$B$ crystal graphs", bg_color='#f8f9fa')

    # ================= 2. 核心 CGCNN 网络数据流 (中间) =================
    ax.text(11, 10.2, "2. Crystal Graph Convolutional Neural Network (Forward Pass)", fontsize=18, weight='bold', ha='center', color='#1864ab')
    # 扩大 CGCNN 外框范围
    cgcnn_box = patches.Rectangle((5.8, 0.8), 10.2, 8.8, linewidth=2, edgecolor='#12b886', facecolor='none', linestyle='-.')
    ax.add_patch(cgcnn_box)
    
    # 输入到图构建
    draw_arrow(5.1, 3.6, 6.2, 3.6, r"CIF $\rightarrow$ Graph", offset_y=0.25)
    
    # Atom & Edge Initialization
    draw_box(6.2, 7.5, 7.5, 1.5, 
             r"Graph Construction $\mathcal{G} = (\mathcal{V}, \mathcal{E})$" + "\n" + 
             r"Atom Nodes $\mathbf{v}_i^{(0)} \in \mathbb{R}^{64}$ (One-hot)" + "\n" + 
             r"Edge Bonds $\mathbf{u}_{i,j} \in \mathbb{R}^{41}$ (Gaussian)", bg_color='#f3f0ff')
    
    # 箭头向下，标注初始张量
    draw_arrow(9.2, 7.5, 9.2, 6.1, r"Tensor: $(B \times N_{atom}, 64)$", offset_y=-0.25)
    draw_arrow(11.5, 7.5, 11.5, 6.1, r"Edge: $(B \times N_{edge}, 41)$", offset_y=-0.25)
    
    # Graph Convolutions
    conv_text = (
        r"Graph Conv Layers ($l=1 \dots 4$)" + "\n\n" +
        r"$\mathbf{z}_{i,j}^{(l)} = \mathbf{v}_i^{(l)} \oplus \mathbf{v}_j^{(l)} \oplus \mathbf{u}_{i,j}$" + "\n" +
        r"$\mathbf{v}_i^{(l+1)} = \mathbf{v}_i^{(l)} + \sum_{j} \sigma(\mathbf{z}_{i,j}^{(l)}\mathbf{W}_f) \odot g(\mathbf{z}_{i,j}^{(l)}\mathbf{W}_s)$"
    )
    draw_box(6.2, 3.8, 7.5, 2.3, conv_text, bg_color='#e3fafc', font_size=15)
    
    # 箭头向下，标注卷积后的张量维度
    draw_arrow(10.0, 3.8, 10.0, 2.5, r"Node Feature: $(B \times N_{atom}, 64)$", offset_y=-0.3)
    
    # Pooling Layer
    draw_box(6.2, 1.3, 7.5, 1.2, 
             r"Normalized Softplus Pooling" + "\n" + 
             r"$\mathbf{v}_c = \sum_{i} \mathbf{v}_i^{(L)}$ / $N_{atom}$", bg_color='#ffe3e3')
    
    # FC Layer
    draw_arrow(13.7, 1.9, 14.5, 1.9, r"Pool: $(B, 64)$", offset_y=0.3)
    
    # FC 竖向大字号
    draw_box(14.5, 3.5, 1.2, 4.5, 
             r"F" + "\n" + r"C" + "\n\n" + r"L" + "\n" + r"A" + "\n" + r"Y" + "\n" + r"E" + "\n" + r"R" + "\n" + r"S", 
             bg_color='#fff0f6', font_size=16)
    
    # FC 内部细节说明 (字体调大)
    ax.text(14.0, 6.5, r"$\mathbb{R}^{64} \rightarrow \mathbb{R}^{128}$", fontsize=12, color='#862e9c')
    ax.text(14.0, 5.7, r"ReLU", fontsize=12, color='#862e9c')
    ax.text(14.0, 4.9, r"$\mathbb{R}^{128} \rightarrow \mathbb{R}^{2}$", fontsize=12, color='#862e9c')
    
    # 单模型输出
    draw_arrow(15.7, 6.5, 17.0, 6.5, r"LogSoftmax: $(B, 2)$", offset_y=0.3)
    draw_box(17.0, 5.8, 3.8, 1.4, r"Single Model Prediction" + "\n" + r"$\hat{y}^{(t)} = P(y=1 | \mathcal{G}, \theta^{(t)})$", bg_color='#f1f3f5')

    # ================= 3. 集成评估 (右侧) =================
    ax.text(18.9, 10.2, "3. Ensemble Output", fontsize=18, weight='bold', ha='center', color='#1864ab')
    
    draw_arrow(18.9, 5.8, 18.9, 4.5)
    
    ensemble_text = (
        r"Crystal-Likeness Score" + "\n\n" +
        r"$CLscore = \frac{1}{T} \sum_{t=1}^{T} \hat{y}^{(t)}$"
    )
    draw_box(17.0, 2.7, 3.8, 1.8, ensemble_text, bg_color='#d8f5a2', edge_color='#5c940d', font_size=15)
    
    draw_arrow(18.9, 2.7, 18.9, 1.7)
    draw_box(17.0, 0.7, 3.8, 1.0, r"Decision Boundary" + "\n" + r"$CLscore > 0.5$", bg_color='#f8f9fa')

    # 全局大标题
    plt.text(10.5, 11.2, "PU-CGCNN: Tensor Flow & Mathematical Architecture", 
             fontsize=26, ha='center', va='center', weight='bold', color='#212529')

    # 保存极高分辨率图像
    plt.tight_layout()
    output_file = os.path.join(r"E:\pu-cgcnn", "Academic_Architecture_Diagram_LargeText.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n[*] 大字号且极其清晰的学术架构图已生成！")
    print(f"[*] 请去检查文件: {output_file}")
    
if __name__ == "__main__":
    draw_academic_architecture_large("")