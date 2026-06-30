# -*- coding: utf-8 -*-
"""
加速版月季花叶片检测 - 两阶段 GrabCut（下采样加速 + 细分割外扩边框）
可视化：原图、粗分割标签、细分割初始化标签、原图轮廓框
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import sys
import time

# 导入自定义自适应 GrabCut
from cv_utils import grabCut_adaptive

# =============================================================================
# 解决中文显示乱码（增加备选字体）
# =============================================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 辅助函数：将 GrabCut 标签矩阵转为彩色图像
# =============================================================================
def label_to_color(label_mask):
    """
    将 GrabCut 标签矩阵转换为彩色图像（RGB）
    颜色映射：
        GC_BGD    (0) → 黑色 (0,0,0)
        GC_FGD    (1) → 白色 (255,255,255)
        GC_PR_BGD (2) → 灰色 (128,128,128)
        GC_PR_FGD (3) → 绿色 (0,255,0)
    """
    color_map = {
        0: (0, 0, 0),          # 确定背景
        1: (255, 255, 255),    # 确定前景
        2: (128, 128, 128),    # 可能背景
        3: (0, 255, 0)         # 可能前景
    }
    h, w = label_mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in color_map.items():
        color_img[label_mask == val] = color
    return color_img

# =============================================================================
# 叶片检测器类（加速版）
# =============================================================================

class LeafDetectorAccelerated:
    """
    快速叶片检测器，使用两阶段 GrabCut。
    内部不包含图像缩放，输入图像尺寸即为输出掩膜尺寸。
    """

    # GrabCut 背景/前景 GMM 模型（类属性，所有实例共享）
    BGD = np.zeros((1, 65), np.float64)
    FGD = np.zeros((1, 65), np.float64)

    def __init__(self):
        """初始化检测结果属性。"""
        self.mask = None          # 最终二值掩膜（与输入图像同尺寸）
        self.contour = None       # 叶片轮廓
        self.area = 0             # 面积（像素）
        self.centroid = (0, 0)    # 质心 (x, y)
        self.coarse_mask = None   # 粗分割二值掩膜（用于可视化）
        self.iters_coarse = 0     # 粗分割迭代次数（固定为1）
        self.iters_fine = 0       # 细分割实际迭代次数

        # 新增属性：存储粗分割和细分割初始化的标签矩阵（用于可视化）
        self.raw_coarse_label = None   # 粗分割标签矩阵（gc）
        self.raw_init_label = None     # 细分割初始化标签矩阵（裁剪后的 gc2）

    def detect(self, img):
        """
        在给定图像中检测叶片区域。

        参数:
            img (np.ndarray): BGR 图像，任意尺寸。

        返回:
            tuple: (leaf_mask, contour) 二元组。
                   - leaf_mask: 二值掩码（255=叶片，0=背景），与输入同尺寸。
                   - contour: 叶片轮廓点数组（无轮廓时为 None）。
        """
        h, w = img.shape[:2]

        # --------------------------------------------------------------------
        # 第一阶段：粗分割（矩形初始化，固定 1 次迭代）
        # --------------------------------------------------------------------
        rect = (int(w * 0.05), int(h * 0.05), int(w * 0.90), int(h * 0.90))
        gc = np.zeros((h, w), np.uint8)
        cv2.grabCut(img, gc, rect, self.BGD, self.FGD, 1, cv2.GC_INIT_WITH_RECT)
        self.iters_coarse = 1
        self.raw_coarse_label = gc.copy()   # 保存粗分割标签矩阵

        # 提取最大连通域作为粗分割叶片
        body = self._largest_component((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD))
        if body is None:
            # 粗分割失败，回退到颜色分割
            fallback = self._color_fallback(img)
            self.coarse_mask = fallback
            # 回退时无法生成标签矩阵，置为 None
            self.raw_coarse_label = None
            return self._from_mask(fallback)

        self.coarse_mask = body.copy()

        # --------------------------------------------------------------------
        # 第二阶段：细分割（mask 模式，外扩 2px 边框 + 调用 grabCut_adaptive）
        # --------------------------------------------------------------------
        # 1) 计算原图质心
        M = cv2.moments(body)
        if M['m00'] == 0:
            return self._from_mask(body)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # 2) 图像外扩 2 像素（复制边界），并计算新尺寸
        pad = 5
        img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        h_pad, w_pad = img_pad.shape[:2]

        # 3) 在扩展图上构建细分割初始化掩膜
        gc2_pad = np.full((h_pad, w_pad), cv2.GC_PR_BGD, dtype=np.uint8)

        # (3.1) 最外一圈 2 像素设为确定背景（GC_BGD）
        gc2_pad[:, :pad] = cv2.GC_BGD
        gc2_pad[:, -pad:] = cv2.GC_BGD
        gc2_pad[:pad, :] = cv2.GC_BGD
        gc2_pad[-pad:, :] = cv2.GC_BGD

        # (3.2) 将粗分割区域（偏移 pad 像素）设为可能前景（GC_PR_FGD）
        gc2_pad[pad:pad + h, pad:pad + w][body > 0] = cv2.GC_PR_FGD

        # (3.3) 质心种子区域（面积 5%，最小 5px）设为确定前景（GC_FGD）
        body_area = cv2.countNonZero(body)
        seed_r = max(int(np.sqrt(body_area) * 0.05), 5)
        # 在扩展图中，种子区域坐标整体 + pad
        gc2_pad[pad + cy - seed_r : pad + cy + seed_r,
                pad + cx - seed_r : pad + cx + seed_r] = cv2.GC_FGD
        
        # 4) 裁剪掉外扩的 2 像素边框，恢复原图尺寸
        gc2 = gc2_pad[pad:-pad, pad:-pad]
        self.raw_init_label = gc2.copy()   # 保存细分割初始化标签矩阵（裁剪后）

        # 5) 调用自适应 GrabCut（在扩展图上运行）
        self.iters_fine = grabCut_adaptive(
            img_pad, gc2_pad, None, self.BGD, self.FGD,
            max_iters=50, tol=0.005, mode=cv2.GC_INIT_WITH_MASK
        )

        gc2 = gc2_pad[pad:-pad, pad:-pad]
        
        # --------------------------------------------------------------------
        # 提取最终前景
        # --------------------------------------------------------------------
        leaf = self._largest_component(
            (gc2 == cv2.GC_FGD) | (gc2 == cv2.GC_PR_FGD)
        )
        if leaf is None:
            return self._from_mask(body)

        # 面积保护：若细分割面积过小（< 总面积的 2%），回退到粗分割
        if cv2.countNonZero(leaf) < h * w * 0.02:
            return self._from_mask(body)

        # 形态学后处理（可选，但保留可提高鲁棒性）
        leaf = self._postprocess(leaf, h, w)

        return self._from_mask(leaf)

    # --------------------------------------------------------------------------
    # 内部辅助方法
    # --------------------------------------------------------------------------

    def _largest_component(self, binary):
        """提取二值图像中面积最大的连通域。"""
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), 8)
        if num <= 1:
            return None
        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        out = np.zeros_like(binary, dtype=np.uint8)
        out[labels == idx] = 255
        return out

    def _from_mask(self, mask):
        """从二值掩码提取轮廓、面积、质心，并存储属性。"""
        if mask is None:
            return self._stash(np.zeros((1, 1), np.uint8), None, 0, (0, 0))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._stash(mask, None, 0, (0, 0))

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.countNonZero(mask)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroid = (cx, cy)
        else:
            x, y, bw, bh = cv2.boundingRect(cnt)
            centroid = (x + bw // 2, y + bh // 2)

        return self._stash(mask, cnt, area, centroid)

    def _stash(self, mask, contour, area, centroid):
        self.mask = mask
        self.contour = contour
        self.area = area
        self.centroid = centroid
        return mask, contour

    def _color_fallback(self, img):
        """HSV 绿色范围提取作为 GrabCut 失败的回退。"""
        h, w = img.shape[:2]
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        mask = cv2.inRange(h_ch, 15, 90) & cv2.threshold(s_ch, 30, 255,
                                                          cv2.THRESH_BINARY)[1]

        scale = min(h, w) / 1000.0
        k_small = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(3, int(3 * scale) | 1),) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_small)

        k_large = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(5, int(11 * scale) | 1),) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_large)

        comp = self._largest_component(mask)
        return comp if comp is not None else mask

    def _postprocess(self, mask, h, w):
        """简单的形态学后处理（在原始尺寸上执行）。"""
        scale = min(h, w) / 1000.0
        k_size = max(3, int(5 * scale) | 1)
        k_large_size = max(5, int(7 * scale) | 1)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        mask = cv2.dilate(mask, k, iterations=2)
        mask = cv2.erode(mask, k, iterations=2)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_large_size, k_large_size))
        )
        return mask


# =============================================================================
# 主函数（含下采样/上采样、可视化、计时）
# =============================================================================

def main():
    # ---------- 选择文件 ----------
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="请选择一张紧贴边缘的叶片图片",
        filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        print("未选择文件。")
        sys.exit(0)

    # ---------- 读取并缩放至基准尺寸（最大边 640） ----------
    img = cv2.imread(file_path)
    h, w = img.shape[:2]
    max_size = 640
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    base_h, base_w = img.shape[:2]          # 基准尺寸
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---------- 下采样至 0.5 倍（面积 25%） ----------
    small_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_AREA)
    small_h, small_w = small_img.shape[:2]

    # ---------- 计时并分割 ----------
    start_time = time.time()

    detector = LeafDetectorAccelerated()
    small_mask, _ = detector.detect(small_img)   # 返回小图掩膜

    end_time = time.time()
    total_time = end_time - start_time

    # ---------- 上采样回基准尺寸 ----------
    # 最终二值掩膜
    base_mask = cv2.resize(small_mask, (base_w, base_h),
                           interpolation=cv2.INTER_NEAREST)

    # 粗分割标签矩阵（如果存在）
    coarse_label_base = None
    if detector.raw_coarse_label is not None:
        coarse_label_base = cv2.resize(detector.raw_coarse_label,
                                       (base_w, base_h),
                                       interpolation=cv2.INTER_NEAREST)

    # 细分割初始化标签矩阵（如果存在）
    init_label_base = None
    if detector.raw_init_label is not None:
        init_label_base = cv2.resize(detector.raw_init_label,
                                     (base_w, base_h),
                                     interpolation=cv2.INTER_NEAREST)

    # ---------- 提取轮廓用于可视化 ----------
    contours, _ = cv2.findContours(base_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    img_res = img_rgb.copy()
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(img_res, [cnt], -1, (255, 0, 0), 2)
        x, y, bw, bh = cv2.boundingRect(cnt)
        cv2.rectangle(img_res, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

    # ---------- 生成彩色标签图像 ----------
    # 粗分割彩色图
    coarse_color = None
    if coarse_label_base is not None:
        coarse_color = label_to_color(coarse_label_base)
    # 初始化彩色图
    init_color = None
    if init_label_base is not None:
        init_color = label_to_color(init_label_base)

    # ---------- 可视化 4 个子图 ----------
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 子图 1：原图
    axes[0].imshow(img_rgb)
    axes[0].set_title("原始图像", fontsize=12)
    axes[0].axis("off")

    # 子图 2：粗分割标签
    if coarse_color is not None:
        axes[1].imshow(coarse_color)
        axes[1].set_title("粗分割标签 (黑=确定背景, 白=确定前景,\n灰=可能背景, 绿=可能前景)", fontsize=10)
    else:
        axes[1].text(0.5, 0.5, "粗分割失败\n(回退颜色分割)", ha='center', va='center', fontsize=12)
    axes[1].axis("off")

    # 子图 3：细分割初始化标签
    if init_color is not None:
        axes[2].imshow(init_color)
        axes[2].set_title("细分割初始化标签\n(种子区域已标记)", fontsize=10)
    else:
        axes[2].text(0.5, 0.5, "无初始化标签", ha='center', va='center', fontsize=12)
    axes[2].axis("off")

    # 子图 4：原图+轮廓+矩形框
    axes[3].imshow(img_res)
    axes[3].set_title(
        f"分割结果 (基准尺寸 {base_w}x{base_h})\n"
        f"粗迭代: {detector.iters_coarse} | 细迭代: {detector.iters_fine}\n"
        f"总耗时: {total_time:.4f} 秒",
        fontsize=12
    )
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()
    cv2.destroyAllWindows()

    # ---------- 打印信息 ----------
    print("=" * 50)
    print(f"基准尺寸: {base_w} x {base_h}")
    print(f"小图尺寸: {small_w} x {small_h}")
    print(f"粗分割迭代次数: {detector.iters_coarse}")
    print(f"细分割迭代次数: {detector.iters_fine}")
    print(f"总执行时间: {total_time:.4f} 秒")
    print("=" * 50)


if __name__ == "__main__":
    main()