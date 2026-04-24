# -*- coding: utf-8 -*-
"""
PU-CGCNN 晶体可合成性预测 GUI
==============================

功能
----
1. 选择单个 .cif 文件或包含 .cif 的文件夹作为输入
2. 自动准备工作目录 (复制 CIF / atom_init.json / 生成 id_prop.csv)
3. 调用 data_PU_learning.CIFData 构建晶体图
4. 遍历所有 checkpoint_bag_*.pth.tar,做集成推理
5. 在 GUI 中展示实时日志 / 进度条 / 排序结果表,并支持 CSV 导出

用法
----
首次启动需在界面上指定四个路径:
  - 源代码目录:包含 data_PU_learning.py 和 model_PU_learning.py
                (或其 cgcnn/ / src_code/ 子目录)
  - 模型目录  :包含 checkpoint_bag_{1..N}.pth.tar
  - atom_init.json:元素初始嵌入文件 (PU-CGCNN 仓库根目录带)
  - 工作临时目录:放临时 CIF 副本和 id_prop.csv,可随便指定

以上配置会保存到 ~/.pucgcnn_gui.json,下次自动读取。
"""
from __future__ import annotations

import os
import sys
import csv
import json
import glob
import queue
import shutil
import threading
import traceback
import argparse
import importlib.util
from collections import defaultdict
from typing import List, Dict, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext


# ======================================================================
#  配置持久化
# ======================================================================
CONFIG_FILE = os.path.join(os.path.expanduser('~'), '.pucgcnn_gui.json')


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_config(cfg: dict):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ======================================================================
#  按绝对路径动态加载模块 —— 绕开 `from cgcnn.xxx` 这种包路径依赖
# ======================================================================
def load_module_from_file(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def locate_source_files(src_dir: str) -> Dict[str, str]:
    """在 src_dir 及常见子目录下找 data_PU_learning.py / model_PU_learning.py。"""
    found: Dict[str, str] = {}
    candidates = ['', 'cgcnn', 'src_code', os.path.join('src_code', 'cgcnn')]
    for name in ('data_PU_learning.py', 'model_PU_learning.py'):
        for sub in candidates:
            p = os.path.join(src_dir, sub, name) if sub else os.path.join(src_dir, name)
            if os.path.isfile(p):
                found[name] = p
                break
    return found


# ======================================================================
#  后台推理线程
# ======================================================================
class PredictionWorker(threading.Thread):
    """在后台线程里完整跑:准备数据 → 构图 → 50 模型集成 → 汇总。"""

    def __init__(self, params: dict, msg_queue: "queue.Queue"):
        super().__init__(daemon=True)
        self.p = params
        self.q = msg_queue
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def log(self, msg: str):
        self.q.put(('log', msg))

    def progress(self, value: int):
        self.q.put(('progress', value))

    def done(self, results: Optional[List[dict]], error: Optional[str] = None):
        self.q.put(('done', results, error))

    # ------------------------------------------------------------------
    def run(self):
        try:
            results = self._run_pipeline()
            self.done(results)
        except Exception as e:
            traceback.print_exc()
            self.done(None, f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    def _run_pipeline(self) -> List[dict]:
        # 惰性 import — GUI 启动时即使没装 torch 也不会崩
        import numpy as np
        import torch
        from torch.utils.data import DataLoader

        p = self.p

        # -------- 1. 准备工作目录 --------
        self.log("=" * 60)
        self.log("阶段 1/4  准备工作目录")
        work_cifs = p['work_cifs_dir']
        os.makedirs(work_cifs, exist_ok=True)

        # 清空旧 .cif / .pickle / id_prop.csv 避免串数据
        for pat in ('*.cif', '*.pickle', 'id_prop.csv'):
            for old in glob.glob(os.path.join(work_cifs, pat)):
                try:
                    os.remove(old)
                except Exception:
                    pass

        input_path = p['input_path']
        if os.path.isfile(input_path):
            if not input_path.lower().endswith('.cif'):
                raise ValueError(f"{input_path} 不是 .cif 文件")
            shutil.copy(input_path, work_cifs)
            self.log(f"  复制单个 CIF 到 {work_cifs}")
        elif os.path.isdir(input_path):
            cif_src = sorted(glob.glob(os.path.join(input_path, '*.cif')))
            if not cif_src:
                raise ValueError(f"{input_path} 下没有 .cif 文件")
            for c in cif_src:
                shutil.copy(c, work_cifs)
            self.log(f"  复制 {len(cif_src)} 个 CIF 到 {work_cifs}")
        else:
            raise ValueError(f"输入路径不存在: {input_path}")

        # atom_init.json 必须和 CIF 放一起 (CIFData 会从 root_dir 读)
        atom_src = p['atom_init_path']
        if not os.path.isfile(atom_src):
            raise FileNotFoundError(f"atom_init.json 不存在: {atom_src}")
        shutil.copy(atom_src, os.path.join(work_cifs, 'atom_init.json'))

        # 写 id_prop.csv (预测不需要真实标签,填 0 即可)
        cifs = sorted(glob.glob(os.path.join(work_cifs, '*.cif')))
        id_prop = os.path.join(work_cifs, 'id_prop.csv')
        with open(id_prop, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            for c in cifs:
                w.writerow([os.path.basename(c)[:-4], 0])
        self.log(f"  写入 id_prop.csv:{len(cifs)} 条")

        # -------- 2. 加载 PU-CGCNN 源码 --------
        self.log("")
        self.log("阶段 2/4  动态加载模型源代码")
        found = locate_source_files(p['src_dir'])
        if 'data_PU_learning.py' not in found or 'model_PU_learning.py' not in found:
            raise FileNotFoundError(
                f"在 {p['src_dir']} (含 cgcnn/、src_code/ 子目录) 下"
                f"未找到 data_PU_learning.py / model_PU_learning.py"
            )
        data_mod = load_module_from_file('pucgcnn_data', found['data_PU_learning.py'])
        model_mod = load_module_from_file('pucgcnn_model', found['model_PU_learning.py'])
        CIFData = data_mod.CIFData
        collate_pool = data_mod.collate_pool
        CrystalGraphConvNet = model_mod.CrystalGraphConvNet
        self.log(f"  data:  {found['data_PU_learning.py']}")
        self.log(f"  model: {found['model_PU_learning.py']}")

        # -------- 3. 逐个构图 (单线程,避开 Windows multiprocessing 死锁) --------
        self.log("")
        self.log("阶段 3/4  构建晶体图")
        dataset = CIFData(work_cifs)
        n_total = len(dataset)
        self.log(f"  数据集规模:{n_total}")

        graphs: List[tuple] = []
        failed: List[tuple] = []
        for i in range(n_total):
            if self._stop_event.is_set():
                self.log("  用户请求中断")
                return []
            cid = dataset.id_prop_data[i][0]
            try:
                graphs.append(dataset[i])
            except Exception as e:
                failed.append((cid, str(e)))
                self.log(f"  ⚠ {cid}: {e}")

            done_ct = i + 1
            if done_ct % 10 == 0 or done_ct == n_total:
                self.log(f"  进度 {done_ct}/{n_total}")
            self.progress(int(30 * done_ct / n_total))

        if not graphs:
            raise RuntimeError("全部晶体图构建失败,无法继续预测")
        self.log(f"  成功 {len(graphs)} / {n_total},失败 {len(failed)}")

        # -------- 4. 扫描 checkpoint,逐个模型推理 --------
        self.log("")
        self.log("阶段 4/4  加载模型并打分")
        model_dir = p['model_dir']
        ckpts = sorted(glob.glob(os.path.join(model_dir, 'checkpoint_bag_*.pth.tar')))
        if not ckpts:
            raise FileNotFoundError(f"{model_dir} 下未发现 checkpoint_bag_*.pth.tar")
        max_bags = p.get('max_bags', 0)
        if max_bags and max_bags > 0:
            ckpts = ckpts[:max_bags]
        self.log(f"  将使用 {len(ckpts)} 个 checkpoint")

        use_cuda = torch.cuda.is_available() and not p.get('force_cpu', False)
        device = torch.device('cuda' if use_cuda else 'cpu')
        self.log(f"  推理设备:{device}")

        # 第一张图决定输入维度 (92 for atom, 41 for gaussian expansion by default)
        (atom_fea0, nbr_fea0, _), _, _ = graphs[0]
        orig_atom_fea_len = atom_fea0.shape[-1]
        nbr_fea_len = nbr_fea0.shape[-1]
        self.log(f"  atom_fea_len={orig_atom_fea_len}, nbr_fea_len={nbr_fea_len}")

        all_preds: Dict[str, List[float]] = defaultdict(list)
        batch_size = int(p.get('batch_size', 64))

        for bag_idx, ckpt_path in enumerate(ckpts):
            if self._stop_event.is_set():
                self.log("  用户请求中断")
                break

            # PyTorch 2.6+ 默认 weights_only=True 会拒绝 argparse.Namespace,强制关闭
            try:
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            except TypeError:
                # 老版本 torch 没有 weights_only 参数
                ckpt = torch.load(ckpt_path, map_location=device)

            margs = argparse.Namespace(**ckpt['args']) if isinstance(ckpt.get('args'), dict) \
                else ckpt['args']

            model = CrystalGraphConvNet(
                orig_atom_fea_len, nbr_fea_len,
                atom_fea_len=getattr(margs, 'atom_fea_len', 64),
                n_conv=getattr(margs, 'n_conv', 3),
                h_fea_len=getattr(margs, 'h_fea_len', 128),
                n_h=getattr(margs, 'n_h', 1),
                classification=True,
            ).to(device)
            model.load_state_dict(ckpt['state_dict'])
            model.eval()

            loader = DataLoader(
                graphs, batch_size=batch_size, shuffle=False,
                collate_fn=collate_pool, num_workers=0,
            )
            with torch.no_grad():
                for (inp, _tgt, cif_ids) in loader:
                    atom_fea = inp[0].to(device, non_blocking=True)
                    nbr_fea = inp[1].to(device, non_blocking=True)
                    nbr_fea_idx = inp[2].to(device, non_blocking=True)
                    cidx = [x.to(device, non_blocking=True) for x in inp[3]]
                    out = model(atom_fea, nbr_fea, nbr_fea_idx, cidx)
                    probs = torch.exp(out).cpu().numpy()  # (B, 2)
                    for k, cid in enumerate(cif_ids):
                        all_preds[cid].append(float(probs[k, 1]))

            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            self.log(f"  [{bag_idx + 1}/{len(ckpts)}] {os.path.basename(ckpt_path)}")
            self.progress(30 + int(70 * (bag_idx + 1) / len(ckpts)))

        # -------- 5. 集成 --------
        results = []
        for cid, probs in all_preds.items():
            arr = np.array(probs)
            results.append({
                'Material_ID': cid,
                'CLscore': float(arr.mean()),
                'StdDev': float(arr.std()),
                'Num_Models': len(probs),
            })
        results.sort(key=lambda r: r['CLscore'], reverse=True)

        self.log("")
        self.log(f"完成!共对 {len(results)} 个材料打分")
        self.progress(100)
        return results


# ======================================================================
#  GUI
# ======================================================================
class PUCGCNNGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PU-CGCNN 晶体可合成性预测")
        self.root.geometry('1150x850')

        self.cfg = load_config()
        self.worker: Optional[PredictionWorker] = None
        self.msg_queue: queue.Queue = queue.Queue()
        self.results: List[dict] = []
        self._last_sort_col: Optional[str] = None

        self._build_ui()
        self._poll_queue()

    # ---------------- UI 搭建 ----------------
    def _build_ui(self):
        pad = {'padx': 6, 'pady': 3}

        # --- 1. 输入 ---
        sec_in = ttk.LabelFrame(self.root, text="1. 输入 CIF")
        sec_in.pack(fill='x', padx=10, pady=5)

        self.input_mode = tk.StringVar(value=self.cfg.get('input_mode', 'folder'))
        row = ttk.Frame(sec_in); row.pack(fill='x', **pad)
        ttk.Radiobutton(row, text='单个 CIF 文件', variable=self.input_mode, value='file')\
            .pack(side='left', padx=5)
        ttk.Radiobutton(row, text='CIF 文件夹', variable=self.input_mode, value='folder')\
            .pack(side='left', padx=5)

        self.input_path = tk.StringVar(value=self.cfg.get('input_path', ''))
        row = ttk.Frame(sec_in); row.pack(fill='x', **pad)
        ttk.Label(row, text="路径:", width=14).pack(side='left')
        ttk.Entry(row, textvariable=self.input_path).pack(side='left', fill='x', expand=True, padx=4)
        ttk.Button(row, text="浏览...", command=self._pick_input).pack(side='left')

        # --- 2. 代码 / 模型 ---
        sec_cfg = ttk.LabelFrame(self.root, text="2. 代码与模型路径 (首次设置后自动记住)")
        sec_cfg.pack(fill='x', padx=10, pady=5)

        self.src_dir = tk.StringVar(value=self.cfg.get('src_dir', ''))
        self.model_dir = tk.StringVar(value=self.cfg.get('model_dir', ''))
        self.atom_init_path = tk.StringVar(value=self.cfg.get('atom_init_path', ''))
        self.work_cifs_dir = tk.StringVar(
            value=self.cfg.get('work_cifs_dir', os.path.join(os.path.expanduser('~'), 'pu_cgcnn_work'))
        )

        for label, var, picker in [
            ("源代码目录",      self.src_dir,        self._pick_src_dir),
            ("模型目录",        self.model_dir,      self._pick_model_dir),
            ("atom_init.json", self.atom_init_path, self._pick_atom_init),
            ("工作临时目录",    self.work_cifs_dir,  self._pick_work_dir),
        ]:
            row = ttk.Frame(sec_cfg); row.pack(fill='x', **pad)
            ttk.Label(row, text=label + ":", width=14).pack(side='left')
            ttk.Entry(row, textvariable=var).pack(side='left', fill='x', expand=True, padx=4)
            ttk.Button(row, text="浏览...", command=picker).pack(side='left')

        # 数值参数
        row = ttk.Frame(sec_cfg); row.pack(fill='x', **pad)
        ttk.Label(row, text="使用模型数 (0=全部):").pack(side='left')
        self.max_bags_var = tk.StringVar(value=str(self.cfg.get('max_bags', 0)))
        ttk.Entry(row, textvariable=self.max_bags_var, width=6).pack(side='left', padx=4)

        ttk.Label(row, text="  Batch size:").pack(side='left')
        self.batch_size_var = tk.StringVar(value=str(self.cfg.get('batch_size', 64)))
        ttk.Entry(row, textvariable=self.batch_size_var, width=6).pack(side='left', padx=4)

        self.force_cpu_var = tk.BooleanVar(value=self.cfg.get('force_cpu', False))
        ttk.Checkbutton(row, text='强制使用 CPU', variable=self.force_cpu_var)\
            .pack(side='left', padx=12)

        ttk.Label(row, text="  高潜力阈值:").pack(side='left')
        self.threshold_var = tk.DoubleVar(value=self.cfg.get('threshold', 0.5))
        ttk.Entry(row, textvariable=self.threshold_var, width=6).pack(side='left', padx=4)

        # --- 3. 控制按钮 ---
        sec_run = ttk.LabelFrame(self.root, text="3. 运行")
        sec_run.pack(fill='x', padx=10, pady=5)

        row = ttk.Frame(sec_run); row.pack(fill='x', **pad)
        self.btn_run = ttk.Button(row, text="▶ 开始预测", command=self.on_run)
        self.btn_run.pack(side='left', padx=4)
        self.btn_stop = ttk.Button(row, text="⏹ 停止", command=self.on_stop, state='disabled')
        self.btn_stop.pack(side='left', padx=4)
        self.btn_export = ttk.Button(row, text="💾 导出 CSV", command=self.on_export, state='disabled')
        self.btn_export.pack(side='left', padx=4)
        self.btn_clear = ttk.Button(row, text="🗑 清空结果", command=self.on_clear)
        self.btn_clear.pack(side='left', padx=4)
        ttk.Button(row, text="ℹ 关于", command=self.show_about).pack(side='right', padx=4)

        self.progress_var = tk.IntVar(value=0)
        self.progressbar = ttk.Progressbar(sec_run, variable=self.progress_var, maximum=100)
        self.progressbar.pack(fill='x', padx=8, pady=(2, 8))

        # --- 4. 日志 + 结果 (左右分栏) ---
        paned = ttk.PanedWindow(self.root, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=10, pady=5)

        # 日志
        sec_log = ttk.LabelFrame(paned, text="4. 运行日志")
        self.log_text = scrolledtext.ScrolledText(
            sec_log, wrap='word', height=22, font=('Consolas', 9)
        )
        self.log_text.pack(fill='both', expand=True, padx=4, pady=4)
        paned.add(sec_log, weight=1)

        # 结果表
        sec_res = ttk.LabelFrame(paned, text="5. 预测结果 (点击表头可排序)")
        cols = ('rank', 'material', 'clscore', 'std', 'n_models', 'flag')
        col_conf = [
            ('rank',     '#',           50),
            ('material', 'Material ID', 170),
            ('clscore',  'CLscore',     100),
            ('std',      'StdDev',      80),
            ('n_models', '#Models',     70),
            ('flag',     '高潜力',       70),
        ]
        tree_frame = ttk.Frame(sec_res)
        tree_frame.pack(fill='both', expand=True, padx=4, pady=4)
        self.tree = ttk.Treeview(tree_frame, columns=cols, show='headings', height=20)
        for c, txt, w in col_conf:
            self.tree.heading(c, text=txt, command=lambda cc=c: self._sort_tree(cc))
            self.tree.column(c, width=w, anchor='center')
        vsb = ttk.Scrollbar(tree_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')
        self.tree.tag_configure('high', background='#d4edda')   # 绿
        self.tree.tag_configure('med',  background='#fff3cd')   # 黄
        paned.add(sec_res, weight=1)

        # 状态栏
        self.status = tk.StringVar(value="就绪")
        tk.Label(self.root, textvariable=self.status, bd=1, relief='sunken', anchor='w')\
            .pack(side='bottom', fill='x')

    # ---------------- 文件 / 文件夹选择 ----------------
    def _pick_input(self):
        if self.input_mode.get() == 'file':
            path = filedialog.askopenfilename(
                title="选择 CIF 文件",
                filetypes=[("CIF files", "*.cif"), ("All", "*.*")])
        else:
            path = filedialog.askdirectory(title="选择 CIF 文件夹")
        if path:
            self.input_path.set(path)

    def _pick_src_dir(self):
        p = filedialog.askdirectory(title="选择包含 data_PU_learning.py / model_PU_learning.py 的目录")
        if p: self.src_dir.set(p)

    def _pick_model_dir(self):
        p = filedialog.askdirectory(title="选择包含 checkpoint_bag_*.pth.tar 的目录")
        if p: self.model_dir.set(p)

    def _pick_atom_init(self):
        p = filedialog.askopenfilename(
            title="选择 atom_init.json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if p: self.atom_init_path.set(p)

    def _pick_work_dir(self):
        p = filedialog.askdirectory(title="选择工作临时目录")
        if p: self.work_cifs_dir.set(p)

    # ---------------- 运行 ----------------
    def on_run(self):
        # --- 校验输入 ---
        input_path = self.input_path.get().strip()
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("输入错误", "请选择存在的 CIF 文件或文件夹")
            return
        src_dir = self.src_dir.get().strip()
        if not src_dir or not os.path.isdir(src_dir):
            messagebox.showerror("配置错误", "请指定源代码目录")
            return
        if not locate_source_files(src_dir):
            messagebox.showerror(
                "源代码未找到",
                f"{src_dir} (及其 cgcnn/ / src_code/ 子目录) 中\n"
                f"未找到 data_PU_learning.py / model_PU_learning.py")
            return
        model_dir = self.model_dir.get().strip()
        if not model_dir or not os.path.isdir(model_dir):
            messagebox.showerror("配置错误", "请指定模型目录")
            return
        if not glob.glob(os.path.join(model_dir, 'checkpoint_bag_*.pth.tar')):
            messagebox.showerror("无模型", f"{model_dir} 下没有 checkpoint_bag_*.pth.tar")
            return
        atom_init = self.atom_init_path.get().strip()
        if not atom_init or not os.path.isfile(atom_init):
            # 尝试在源代码目录里找
            for guess in (os.path.join(src_dir, 'atom_init.json'),
                          os.path.join(os.path.dirname(input_path), 'atom_init.json')):
                if os.path.isfile(guess):
                    atom_init = guess
                    self.atom_init_path.set(guess)
                    break
            else:
                messagebox.showerror("缺少 atom_init.json", "请指定 atom_init.json 文件")
                return

        try:
            max_bags = int(self.max_bags_var.get() or 0)
            batch_size = int(self.batch_size_var.get() or 64)
            threshold = float(self.threshold_var.get())
        except ValueError:
            messagebox.showerror("参数错误", "模型数 / Batch size 需是整数,阈值需是数字")
            return

        work_dir = self.work_cifs_dir.get().strip() or \
            os.path.join(os.path.expanduser('~'), 'pu_cgcnn_work')

        # --- 持久化 ---
        self.cfg.update({
            'input_mode': self.input_mode.get(),
            'input_path': input_path,
            'src_dir': src_dir,
            'model_dir': model_dir,
            'atom_init_path': atom_init,
            'work_cifs_dir': work_dir,
            'max_bags': max_bags,
            'batch_size': batch_size,
            'force_cpu': self.force_cpu_var.get(),
            'threshold': threshold,
        })
        save_config(self.cfg)

        # --- UI 重置 ---
        self.on_clear(confirm=False)
        self.btn_run.config(state='disabled')
        self.btn_export.config(state='disabled')
        self.btn_stop.config(state='normal')
        self.status.set("运行中...")

        # --- 启动线程 ---
        params = {
            'input_path': input_path,
            'src_dir': src_dir,
            'model_dir': model_dir,
            'atom_init_path': atom_init,
            'work_cifs_dir': os.path.join(work_dir, 'cifs'),
            'max_bags': max_bags,
            'batch_size': batch_size,
            'force_cpu': self.force_cpu_var.get(),
        }
        self.worker = PredictionWorker(params, self.msg_queue)
        self.worker.start()

    def on_stop(self):
        if self.worker and self.worker.is_alive():
            self.worker.stop()
            self.status.set("正在停止 (完成当前 bag 后退出)...")

    def on_clear(self, confirm=True):
        if confirm and self.tree.get_children():
            if not messagebox.askyesno("确认", "清空日志和结果表?"):
                return
        self.results = []
        self.tree.delete(*self.tree.get_children())
        self.log_text.delete('1.0', 'end')
        self.progress_var.set(0)

    def on_export(self):
        if not self.results:
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[("CSV", "*.csv")],
            title="保存结果为 CSV",
            initialfile='pu_cgcnn_results.csv',
        )
        if not path:
            return
        try:
            # utf-8-sig 让 Excel 直接识别中文表头
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(['Rank', 'Material_ID', 'CLscore', 'StdDev', 'Num_Models'])
                for i, r in enumerate(self.results, 1):
                    w.writerow([i, r['Material_ID'],
                                f"{r['CLscore']:.6f}",
                                f"{r['StdDev']:.6f}",
                                r['Num_Models']])
            messagebox.showinfo("导出成功", f"已保存到:\n{path}")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))

    def show_about(self):
        messagebox.showinfo(
            "关于",
            "PU-CGCNN 晶体可合成性预测 GUI\n"
            "-----------------------------------\n"
            "流程: CIF → pymatgen 构图 → \n"
            "      N 个 CGCNN 子模型推理 → \n"
            "      概率均值集成 → CLscore (0~1)\n"
            "      越高越可能可合成\n\n"
            "算法: Jang J. et al., 2020,\n"
            "      KAIST Jung Group\n"
            "      Synthesizability-PU-CGCNN"
        )

    # ---------------- 消息队列轮询 ----------------
    def _poll_queue(self):
        try:
            while True:
                item = self.msg_queue.get_nowait()
                kind = item[0]
                if kind == 'log':
                    self._append_log(item[1])
                elif kind == 'progress':
                    self.progress_var.set(item[1])
                elif kind == 'done':
                    _, results, error = item
                    self._on_done(results, error)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _append_log(self, msg: str):
        self.log_text.insert('end', msg + '\n')
        self.log_text.see('end')

    def _on_done(self, results, error):
        self.btn_run.config(state='normal')
        self.btn_stop.config(state='disabled')
        if error:
            self.status.set(f"错误: {error}")
            messagebox.showerror("预测失败", error)
            return
        if not results:
            self.status.set("已停止或无结果")
            return

        self.results = results
        thr = self.threshold_var.get()
        for i, r in enumerate(results, 1):
            if r['CLscore'] >= thr:
                tag = ('high',); flag = '⭐'
            elif r['CLscore'] >= thr * 0.7:
                tag = ('med',);  flag = '·'
            else:
                tag = (); flag = ''
            self.tree.insert('', 'end',
                values=(i, r['Material_ID'],
                        f"{r['CLscore']:.4f}",
                        f"{r['StdDev']:.4f}",
                        r['Num_Models'], flag),
                tags=tag)
        self.btn_export.config(state='normal')
        high_ct = sum(1 for r in results if r['CLscore'] >= thr)
        self.status.set(
            f"完成:{len(results)} 个材料,其中 {high_ct} 个 CLscore ≥ {thr:.2f}"
        )

    def _sort_tree(self, col: str):
        items = [(self.tree.set(k, col), k) for k in self.tree.get_children('')]
        try:
            items.sort(key=lambda x: float(x[0]), reverse=True)
        except ValueError:
            items.sort(reverse=True)
        if self._last_sort_col == col:
            items.reverse()
            self._last_sort_col = None
        else:
            self._last_sort_col = col
        for i, (_, k) in enumerate(items):
            self.tree.move(k, '', i)


# ======================================================================
#  入口
# ======================================================================
def main():
    root = tk.Tk()
    # Windows 下 HiDPI 屏幕防糊
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    # ttk 主题
    try:
        style = ttk.Style(root)
        for theme in ('vista', 'winnative', 'clam', 'alt'):
            if theme in style.theme_names():
                style.theme_use(theme)
                break
    except Exception:
        pass
    PUCGCNNGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
