# -*- coding: utf-8 -*-
"""
叶片检测器三种模式对比测试（缩放到最大边320）
在同一张图像上运行 'seed', 'padding', 'twostage' 三种分割模式，
先缩放到最大边320进行检测，再上采样回原图显示轮廓和矩形框，
并记录执行时间。
"""

import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

from leaf_detector import LeafDetector

# ---------- 中文字体设置 ----------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    # ---------- 1. 选择图片 ----------
    Tk().withdraw()
    img_path = filedialog.askopenfilename(
        title="请选择一张叶片图片",
        filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tif")]
    )
    if not img_path:
        print("未选择图片，程序退出。")
        sys.exit(0)

    # ---------- 2. 读取图片 ----------
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"无法读取图片: {img_path}")
        sys.exit(1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_bgr.shape[:2]
    print(f"原图尺寸: {w_orig} x {h_orig}")

    # ---------- 3. 缩放到最大边 320 ----------
    
    h_img, w_img = img_bgr.shape[:2]
    max_size = 320
    if max(h_img, w_img) > max_size:
        scale = max_size / max(h_img, w_img)
        img_resized = cv2.resize(img_bgr, (int(w_img * scale), int(h_img * scale)),
                                 interpolation=cv2.INTER_AREA)
    else:
        img_resized = img_bgr.copy()
    h_resized, w_resized = img_resized.shape[:2]
    print(f"缩放后尺寸: {w_resized} x {h_resized}")

    # ---------- 4. 三种模式检测（均在缩放后的图像上进行） ----------
    modes = ['seed', 'padding', 'twostage']
    results = []   # 存储 (mode_name, mask_resized, contour_orig, bbox_orig, elapsed)

    for mode in modes:
        print(f"\n正在运行模式: {mode}")
        detector = LeafDetector(mode=mode)
        start = time.time()
        mask_resized = detector.detect(img_resized)   # 在缩放图上运行
        elapsed = time.time() - start
        print(f"  耗时: {elapsed:.4f} 秒")

        # ---------- 5. 将掩膜上采样回原图尺寸（最近邻插值） ----------
        mask_orig = cv2.resize(mask_resized, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        # 提取轮廓和矩形框（在原图上）
        contours, _ = cv2.findContours(mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_cnt)
        else:
            max_cnt = None
            x = y = w = h = 0

        results.append((mode, mask_orig, max_cnt, (x, y, w, h), elapsed))

    # ---------- 6. 可视化比较 ----------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"叶片检测三种模式对比 (原图尺寸: {w_orig}x{h_orig}, 检测尺寸: {w_resized}x{h_resized})", fontsize=14)

    for idx, (mode, mask_orig, cnt, bbox, elapsed) in enumerate(results):
        ax = axes[idx]
        img_marked = img_rgb.copy()
        if cnt is not None:
            cv2.drawContours(img_marked, [cnt], -1, (0, 255, 0), 2)   # 绿色轮廓
            x, y, w, h = bbox
            cv2.rectangle(img_marked, (x, y), (x+w, y+h), (255, 0, 0), 2)  # 红色矩形
            area = np.count_nonzero(mask_orig)
            cv2.putText(img_marked, f"Area: {area} px",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        ax.imshow(img_marked)
        mode_names = {'seed': '种子点引导', 'padding': '复制外边界', 'twostage': '两阶段分割'}
        ax.set_title(f"{mode_names[mode]}\n耗时: {elapsed:.4f} 秒", fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # ---------- 7. 打印详细时间对比 ----------
    print("\n=== 运行时间对比 ===")
    for mode, _, _, _, elapsed in results:
        mode_names = {'seed': '种子点引导', 'padding': '复制外边界', 'twostage': '两阶段分割'}
        print(f"{mode_names[mode]}: {elapsed:.4f} 秒")


if __name__ == "__main__":
    main()
