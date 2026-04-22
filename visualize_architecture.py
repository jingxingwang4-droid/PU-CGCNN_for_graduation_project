import os
import glob
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def extract_model_info(work_dir):
    """从本地读取模型数量和结构参数"""
    bag_files = glob.glob(os.path.join(work_dir, "checkpoint_bag_*.pth.tar"))
    if not bag_files:
        bag_files = glob.glob(os.path.join(work_dir, "model_highest_AUC_bag_*.pth.tar"))
        
    num_models = len(bag_files)
    
    # 默认超参数
    args_info = {
        'atom_fea_len': 64,
        'h_fea_len': 128,
        'n_conv': 4,
        'n_h': 1
    }
    
    if num_models > 0:
        try:
            checkpoint = torch.load(bag_files[0], map_location=torch.device('cpu'))
            if 'args' in checkpoint:
                c_args = checkpoint['args']
                args_info['atom_fea_len'] = getattr(c_args, 'atom_fea_len', 64)
                args_info['h_fea_len'] = getattr(c_args, 'h_fea_len', 128)
                args_info['n_conv'] = getattr(c_args, 'n_conv', 4)
                args_info['n_h'] = getattr(c_args, 'n_h', 1)
        except Exception as e:
            print(f"读取模型超参数失败，使用默认值。({e})")
            
    return num_models, args_info

def draw_architecture(num_models, args_info, output_path):
    """使用 Matplotlib 绘制高分辨率架构图"""
    fig, ax = plt.subplots(figsize=(15, 8), facecolor='#f8f9fa')
    ax.axis('off')
    
    # 【修复核心】强制设定画布视野坐标范围，防止画到屏幕外！
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    
    # 定义绘图辅助函数 (已移除可能引起报错的圆角 rx/ry 参数)
    def draw_box(ax, x, y, width, height, text, bg_color, text_color='white', fontsize=12):
        box = patches.Rectangle((x, y), width, height, linewidth=1.5, edgecolor='#343a40', facecolor=bg_color)
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text, color=text_color, fontsize=fontsize, 
                ha='center', va='center', fontweight='bold')
                
    def draw_arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='#495057', shrink=0.05, width=2, headwidth=8))

    # ================= 1. 数据输入层 =================
    draw_box(ax, 0.5, 6, 3, 1.5, "Database (MP / ICSD)\nPositive & Unlabeled", '#4263eb')
    draw_arrow(ax, 2.0, 5.8, 2.0, 5.0)
    
    draw_box(ax, 0.5, 3.5, 3, 1.5, f"PU Learning\nBagging Sampler\n(1:1 Positive/Negative)", '#f59f00')
    
    # ================= 2. 集成模型层 =================
    draw_arrow(ax, 3.7, 4.25, 4.5, 6.0)
    draw_arrow(ax, 3.7, 4.25, 4.5, 4.25)
    draw_arrow(ax, 3.7, 4.25, 4.5, 2.5)
    
    # 画三个代表性的模型框
    draw_box(ax, 4.7, 5.5, 3.5, 1, "CGCNN Model 1", '#12b886')
    draw_box(ax, 4.7, 3.75, 3.5, 1, "CGCNN Model 2", '#12b886')
    ax.text(6.45, 3.25, "......", fontsize=20, ha='center', color='#868e96')
    draw_box(ax, 4.7, 2.0, 3.5, 1, f"CGCNN Model {num_models}", '#12b886')
    
    # ================= 3. CGCNN 内部架构放大 =================
    cgcnn_box = patches.Rectangle((8.8, 1.5), 4.5, 5.5, linewidth=2, edgecolor='#12b886', facecolor='none', linestyle='--')
    ax.add_patch(cgcnn_box)
    ax.text(11.05, 7.2, "CGCNN Architecture Detail", fontsize=14, ha='center', color='#08a045', fontweight='bold')
    
    # 内部层
    draw_box(ax, 9.2, 5.8, 3.7, 0.8, f"Atom Embedding\n(Len: {args_info['atom_fea_len']})", '#eebefa', '#5c0fa3')
    draw_arrow(ax, 11.05, 5.6, 11.05, 5.0)
    
    draw_box(ax, 9.2, 4.2, 3.7, 0.8, f"{args_info['n_conv']}x Graph Conv Layers", '#d0ebff', '#0b7285')
    draw_arrow(ax, 11.05, 4.0, 11.05, 3.4)
    
    draw_box(ax, 9.2, 2.6, 3.7, 0.8, "Pooling & FC Layers", '#ffc9c9', '#c92a2a')
    
    # 箭头从 Model 1 指向详情框
    draw_arrow(ax, 8.4, 6.0, 8.7, 6.0)
    
    # ================= 4. 预测输出层 =================
    draw_arrow(ax, 13.5, 4.25, 14.3, 4.25)
    draw_box(ax, 14.5, 3.5, 2.5, 1.5, "Average Vote\n(CLscore)", '#7048e8')
    
    # 标题和注脚
    plt.text(9.0, 9.2, f"PU-CGCNN Ensemble Architecture ({num_models} Models)", 
             fontsize=22, ha='center', va='center', fontweight='bold', color='#212529')
    plt.text(9.0, 0.5, "Generated directly from locally trained .pth.tar checkpoints on RTX 5070 Ti Laptop", 
             fontsize=11, ha='center', color='#868e96', style='italic')

    # 【修复优化】调整布局并适当降低 DPI 减小体积 (200 DPI 足够 PPT 高清展示)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n[*] 漂亮！模型架构图已成功生成并保存至: {output_path}")
    print(f"[*] 包含模型数量: {num_models}")
    print(f"[*] CGCNN卷积层数: {args_info['n_conv']}, 节点特征维度: {args_info['atom_fea_len']}")

if __name__ == "__main__":
    WORK_DIR = r"E:\pu-cgcnn"
    OUTPUT_IMG = os.path.join(WORK_DIR, "PU_CGCNN_Architecture_Report.png")
    
    print("[*] 正在扫描模型文件并提取计算图结构...")
    num, args_info = extract_model_info(WORK_DIR)
    
    if num == 0:
        print("[!] 未找到模型文件，将使用默认参数生成示意图。")
        num = 50 
        
    draw_architecture(num, args_info, OUTPUT_IMG)