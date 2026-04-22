import os
import glob
import argparse
import torch
import numpy as np
# 导入你源码中的 CGCNN 模型类
from src_code.cgcnn.model_PU_learning import CrystalGraphConvNet

def count_parameters(model)
    计算模型的可训练参数总量
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_size(size_bytes)
    字节单位转换
    if size_bytes  1024
        return f{size_bytes} B
    elif size_bytes  1024  1024
        return f{size_bytes  1024.2f} KB
    else
        return f{size_bytes  (1024  1024).2f} MB

def main()
    work_dir = rEpu-cgcnn
    
    print(n + ★60)
    print(   12 + PU-CGCNN 集成模型架构与规模分析报告)
    print(★60 + n)

    # 1. 扫描并统计模型数量与大小
    model_files = glob.glob(os.path.join(work_dir, checkpoint_bag_.pth.tar))
    
    if not model_files
        print([!] 未找到模型文件，请确保路径正确。)
        return

    num_models = len(model_files)
    total_size = sum(os.path.getsize(f) for f in model_files)
    
    print(【1. 集成学习 (Ensemble) 规模概览】)
    print(-  50)
    print(f 📦 发现可用子模型数量  {num_models} 个 (对应 {num_models} 次 Bagging 抽样))
    print(f 💾 模型权重总占用空间  {format_size(total_size)})
    print(f 🎯 预测机制           {num_models} 个模型独立打分后求均值 (CLscore))
    print(-  50 + n)

    # 2. 读取其中一个模型，解析其内部架构
    sample_model_path = model_files[0]
    print(【2. 单一图神经网络 (GNN) 内部架构分析】)
    print(-  50)
    print(f 🔍 正在反序列化采样模型 {os.path.basename(sample_model_path)}...)
    
    try
        # 加载权重与超参数字典 (映射到 CPU 防止显存报错)
        checkpoint = torch.load(sample_model_path, map_location='cpu')
        model_args = argparse.Namespace(checkpoint['args'])
        
        # 晶体图特征的默认维度 (CGCNN 默认 原子初始化长度92, 键长展开长度41)
        orig_atom_fea_len = 92  
        nbr_fea_len = 41        

        # 实例化模型架构 (不加载权重，只看骨架)
        model = CrystalGraphConvNet(
            orig_atom_fea_len, nbr_fea_len,
            atom_fea_len=model_args.atom_fea_len,
            n_conv=model_args.n_conv,
            h_fea_len=model_args.h_fea_len,
            n_h=model_args.n_h,
            classification=True
        )

        params_per_model = count_parameters(model)
        total_ensemble_params = params_per_model  num_models

        print(f ⚙️  超参数配置 图卷积层数={model_args.n_conv}, 隐藏层神经元={model_args.h_fea_len})
        print(f 🧠 单个模型可训练参数量 {params_per_model,} 个)
        print(f 🌍 整个集成系统总参数量 {total_ensemble_params,} 个 (算力怪兽！))
        print(n [⬇️ PyTorch 神经网络拓扑结构 ⬇️])
        print(=  50)
        # 打印模型网络结构
        print(model)
        print(=  50 + n)
        
    except Exception as e
        print(f解析模型架构时出错 {e})

if __name__ == __main__
    main()