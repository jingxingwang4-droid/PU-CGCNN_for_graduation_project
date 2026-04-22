import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_max_architecture(output_path):
    """绘制海报级大字号、极大画布的学术版架构图"""
    # 1. 终极扩展画布：宽度28，高度15 (原先是22x12)
    fig, ax = plt.subplots(figsize=(28, 15), facecolor='white')
    ax.axis('off')
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 15)
    
    # 辅助绘图函数 (默认基础字号飙升至 18)
    def draw_box(x, y, w, h, text, bg_color='#f8f9fa', edge_color='#212529', font_size=18, font_weight='normal'):
        box = patches.Rectangle((x, y), w, h, linewidth=2.0, edgecolor=edge_color, facecolor=bg_color)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, color='#212529', fontsize=font_size, 
                ha='center', va='center', weight=font_weight)

    def draw_arrow(x1, y1, x2, y2, label="", offset_y=0.3, font_size=16):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='#495057', shrink=0.02, width=2.5, headwidth=10))
        if label:
            # 箭头标注字体加大加粗
            ax.text((x1+x2)/2, (y1+y2)/2 + offset_y, label, ha='center', va='center', 
                    fontsize=font_size, color='#d9480f', style='italic', weight='bold')

    # ================= 1. PU Learning 数据采样流 (左侧) =================
    # 列标题字号 24
    ax.text(4.0, 13.5, "1. PU Bagging Strategy", fontsize=24, weight='bold', ha='center', color='#1864ab')
    
    draw_box(1.0, 11.0, 6.0, 1.8, r"Database $\mathcal{D}$" + "\n" + r"Positive $\mathcal{P}$, Unlabeled $\mathcal{U}$", bg_color='#e7f5ff', font_size=20)
    draw_arrow(4.0, 11.0, 4.0, 9.5, r"Subset Extraction")
    
    draw_box(1.0, 6.5, 6.0, 3.0, r"Iteration $t \in \{1, \dots, T\}$" + "\n\n" + 
             r"$\mathcal{D}_{train}^{(t)} = \mathcal{P} \cup \mathcal{U}_n^{(t)}$" + "\n" + 
             r"$|\mathcal{P}| = |\mathcal{U}_n^{(t)}| = K$", bg_color='#fff3bf', font_size=22)
    
    draw_arrow(4.0, 6.5, 4.0, 5.0, r"Mini-Batch Sampling")
    draw_box(1.0, 3.0, 6.0, 2.0, r"Input Batch $\mathcal{B}$" + "\n" + r"$B$ crystal graphs", bg_color='#f8f9fa', font_size=20)

    # ================= 2. 核心 CGCNN 网络数据流 (中间) =================
    ax.text(14.0, 13.5, "2. Crystal Graph Convolutional Neural Network (Forward Pass)", fontsize=24, weight='bold', ha='center', color='#1864ab')
    
    # 极宽的 CGCNN 虚线外框
    cgcnn_box = patches.Rectangle((8.0, 1.0), 12.0, 11.5, linewidth=2.5, edgecolor='#12b886', facecolor='none', linestyle='-.')
    ax.add_patch(cgcnn_box)
    
    # 输入到图构建
    draw_arrow(7.0, 4.0, 8.5, 4.0, r"CIF $\rightarrow$ Graph", offset_y=0.4)
    
    # Atom & Edge Initialization
    draw_box(8.5, 9.8, 9.0, 2.2, 
             r"Graph Construction $\mathcal{G} = (\mathcal{V}, \mathcal{E})$" + "\n" + 
             r"Atom Nodes $\mathbf{v}_i^{(0)} \in \mathbb{R}^{64}$ (One-hot)" + "\n" + 
             r"Edge Bonds $\mathbf{u}_{i,j} \in \mathbb{R}^{41}$ (Gaussian)", bg_color='#f3f0ff', font_size=20)
    
    # 箭头向下
    draw_arrow(11.0, 9.8, 11.0, 8.0, r"Tensor: $(B \times N_{atom}, 64)$", offset_y=-0.3)
    draw_arrow(14.5, 9.8, 14.5, 8.0, r"Edge: $(B \times N_{edge}, 41)$", offset_y=-0.3)
    
    # Graph Convolutions (公式极限放大)
    conv_text = (
        r"Graph Conv Layers ($l=1 \dots 4$)" + "\n\n" +
        r"$\mathbf{z}_{i,j}^{(l)} = \mathbf{v}_i^{(l)} \oplus \mathbf{v}_j^{(l)} \oplus \mathbf{u}_{i,j}$" + "\n\n" +
        r"$\mathbf{v}_i^{(l+1)} = \mathbf{v}_i^{(l)} + \sum_{j} \sigma(\mathbf{z}_{i,j}^{(l)}\mathbf{W}_f) \odot g(\mathbf{z}_{i,j}^{(l)}\mathbf{W}_s)$"
    )
    draw_box(8.5, 4.8, 9.0, 3.2, conv_text, bg_color='#e3fafc', font_size=22)
    
    # 箭头向下
    draw_arrow(13.0, 4.8, 13.0, 3.2, r"Node Feature: $(B \times N_{atom}, 64)$", offset_y=-0.35)
    
    # Pooling Layer
    draw_box(8.5, 1.5, 9.0, 1.7, 
             r"Normalized Softplus Pooling" + "\n" + 
             r"$\mathbf{v}_c = \sum_{i} \mathbf{v}_i^{(L)}$ / $N_{atom}$", bg_color='#ffe3e3', font_size=20)
    
    # FC Layer
    draw_arrow(17.5, 2.3, 18.2, 2.3, r"Pool: $(B, 64)$", offset_y=0.4)
    
    draw_box(18.2, 2.0, 1.3, 7.5, 
             r"F" + "\n" + r"C" + "\n\n" + r"L" + "\n" + r"A" + "\n" + r"Y" + "\n" + r"E" + "\n" + r"R" + "\n" + r"S", 
             bg_color='#fff0f6', font_size=22)
    
    # FC 内部旁注加大
    ax.text(17.6, 8.2, r"$\mathbb{R}^{64} \rightarrow \mathbb{R}^{128}$", fontsize=15, color='#862e9c')
    ax.text(17.6, 6.8, r"ReLU", fontsize=15, color='#862e9c')
    ax.text(17.6, 5.4, r"$\mathbb{R}^{128} \rightarrow \mathbb{R}^{2}$", fontsize=15, color='#862e9c')
    
    # ================= 3. 集成评估 (右侧) =================
    ax.text(24.0, 13.5, "3. Ensemble Output", fontsize=24, weight='bold', ha='center', color='#1864ab')
    
    draw_arrow(19.5, 8.0, 21.5, 8.0, r"LogSoftmax: $(B, 2)$", offset_y=0.4)
    draw_box(21.5, 7.0, 5.5, 2.2, r"Single Model Prediction" + "\n" + r"$\hat{y}^{(t)} = P(y=1 | \mathcal{G}, \theta^{(t)})$", bg_color='#f1f3f5', font_size=20)

    draw_arrow(24.2, 7.0, 24.2, 5.2)
    
    ensemble_text = (
        r"Crystal-Likeness Score" + "\n\n" +
        r"$CLscore = \frac{1}{T} \sum_{t=1}^{T} \hat{y}^{(t)}$"
    )
    draw_box(21.5, 2.5, 5.5, 2.7, ensemble_text, bg_color='#d8f5a2', edge_color='#5c940d', font_size=22)
    
    draw_arrow(24.2, 2.5, 24.2, 1.5)
    draw_box(21.5, 0.5, 5.5, 1.0, r"Decision Boundary" + "\n" + r"$CLscore > 0.5$", bg_color='#f8f9fa', font_size=20)

    # 全局特大标题
    plt.text(14.0, 14.5, "PU-CGCNN: Tensor Flow & Mathematical Architecture", 
             fontsize=36, ha='center', va='center', weight='bold', color='#212529')

    # 保存图像
    plt.tight_layout()
    output_file = os.path.join(r"E:\pu-cgcnn", "Academic_Architecture_Diagram_MAX.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n[*] 巨无霸版架构图已生成！请查看: {output_file}")
    
if __name__ == "__main__":
    draw_max_architecture("")