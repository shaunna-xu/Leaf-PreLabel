# -*- coding: utf-8 -*-
"""
主程序入口
使用 Tkinter 选择文件夹，输入类别 ID，选择分割模式，批量生成 YOLO 标签。
输出路径固定为项目目录下的 outputs/labels/
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog
from batch_YOLO_labeler import BatchYOLOLabeler


if __name__ == "__main__":
    # 1. 弹窗选择文件夹
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="请选择包含叶片图片的文件夹")
    
    if not folder_path:
        print("⚠️ 未选择文件夹，程序退出。")
        sys.exit(0)

    # 2. 输入类别 ID
    try:
        class_id = int(input("请输入该类别对应的 YOLO 编号 (例如 Healthy → 0): ").strip())
    except ValueError:
        print("❌ 类别编号必须是整数，程序退出。")
        sys.exit(1)

    # 3. 选择分割模式（带介绍）
    print("\n" + "="*50)
    print("请选择分割模式：")
    print("="*50)
    print("1. seed (种子点引导)")
    print("   - 原理：OTSU 二值化找质心，构建种子掩膜，单阶段 GrabCut 迭代")
    print("   - 特点：适用于背景简单、叶片与背景对比明显的图片")
    print("-"*50)
    print("2. padding (边缘复制)")
    print("   - 原理：图像边缘向外复制 10 像素，用矩形框包裹原图，单阶段 GrabCut 迭代")
    print("   - 特点：可处理叶片紧贴图像边缘的情况，适用于背景复杂的图片")
    print("-"*50)
    print("3. twostage (两阶段 GrabCut 迭代分割)")
    print("   - 原理：粗分割（矩形留边5%）找质心 → 细分割（掩膜初始化）")
    print("   - 特点：可处理背景复杂的情况")
    print("-"*50)
    print("   - 不同方法执行时间与图片相关")
    print("="*50)

    mode_choice = input("请输入数字选择模式 (1/2/3): ").strip()
    mode_map = {'1': 'seed', '2': 'padding', '3': 'twostage'}
    if mode_choice not in mode_map:
        print("❌ 无效选择，默认使用 'seed' 模式。")
        mode = 'seed'
    else:
        mode = mode_map[mode_choice]
    print(f"✅ 已选择模式: {mode}")

    # ------------------------------------------------------------------------
    # 基于 __file__ 锚定项目根目录
    # ------------------------------------------------------------------------
    src_dir = os.path.dirname(os.path.abspath(__file__))   # 得到 src/ 目录
    project_root = os.path.dirname(src_dir)                # 得到 Leaf-PreLabel 根目录
    
    # 拼接固定的输出路径：项目根目录/outputs/labels
    output_dir = os.path.join(project_root, "outputs", "labels")
    os.makedirs(output_dir, exist_ok=True)

    # 4. 执行批量标注（传入 mode）
    labeler = BatchYOLOLabeler(mode=mode)
    labeler.process_folder(folder_path, class_id, output_dir=output_dir)

    input("\n按 Enter 键退出...")