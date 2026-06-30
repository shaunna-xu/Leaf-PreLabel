# -*- coding: utf-8 -*-
"""
批量自动生成 YOLO 标签（使用两阶段 GrabCut + 下采样加速）
- 用户选择文件夹
- 输入类别 ID
- 对文件夹内所有图片自动分割并输出 YOLO 标签文件
- 输出总耗时
- 该方法可以解决叶片靠近边界和背景复杂的问题
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import time
import sys

# 导入自定义自适应 GrabCut
from cv_utils import grabCut_adaptive


# =============================================================================
# 叶片检测器（简化版，去掉可视化中间存储）
# =============================================================================
class LeafDetectorAccelerated:
    """
    快速叶片检测器，使用两阶段 GrabCut。
    内部不包含图像缩放，输入图像尺寸即为输出掩膜尺寸。
    仅保留必要的结果属性，移除所有可视化用中间变量。
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
        self.iters_coarse = 0     # 粗分割迭代次数（固定为1）
        self.iters_fine = 0       # 细分割实际迭代次数

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

        # 提取最大连通域作为粗分割叶片
        body = self._largest_component((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD))
        if body is None:
            # 粗分割失败，回退到颜色分割
            fallback = self._color_fallback(img)
            return self._from_mask(fallback)

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
        pad = 2
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
        gc2_pad[pad + cy - seed_r : pad + cy + seed_r,
                pad + cx - seed_r : pad + cx + seed_r] = cv2.GC_FGD

        # 4) 调用自适应 GrabCut（在扩展图上运行）
        self.iters_fine = grabCut_adaptive(
            img_pad, gc2_pad, None, self.BGD, self.FGD,
            max_iters=50, tol=0.005, mode=cv2.GC_INIT_WITH_MASK
        )

        # 5) 裁剪掉外扩的 2 像素边框，恢复原图尺寸
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

        # 形态学后处理
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
# 批量生成 YOLO 标签
# =============================================================================
class BatchYOLOLabeler:
    """批量自动标注类，利用 LeafDetectorAccelerated 进行分割。"""

    def __init__(self):
        self.detector = LeafDetectorAccelerated()

    def _image_to_yolo(self, img_path, class_id):
        """对单张图片生成 YOLO 标签字符串（归一化坐标）。"""
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            return None
        h_orig, w_orig = img.shape[:2]

        # 预处理：缩放至最大边 640，再下采样至 0.5 倍（与主函数一致）
        max_size = 640
        if max(h_orig, w_orig) > max_size:
            scale = max_size / max(h_orig, w_orig)
            img = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)))
        # 下采样至 0.5 倍（面积 25%）
        small_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5,
                               interpolation=cv2.INTER_AREA)

        # 调用分割器（返回掩膜）
        small_mask, _ = self.detector.detect(small_img)

        # 上采样回原图尺寸（注意：这里需要上采样回原始分辨率）
        # 但 YOLO 标签需要归一化到原图尺寸，所以上采样到原图尺寸
        mask_orig = cv2.resize(small_mask, (w_orig, h_orig),
                               interpolation=cv2.INTER_NEAREST)

        # 从掩膜提取边界框
        contours, _ = cv2.findContours(mask_orig, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        max_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_cnt)

        # 过滤过小框
        if w < 10 or h < 10:
            return None

        # YOLO 格式：class_id x_center y_center width height (归一化)
        x_center = (x + w / 2.0) / w_orig
        y_center = (y + h / 2.0) / h_orig
        width_norm = w / w_orig
        height_norm = h / h_orig
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"

    def process_folder(self, folder_path, class_id):
        """批量处理文件夹中的所有图片。"""
        if not os.path.exists(folder_path):
            print(f"❌ 文件夹路径不存在: {folder_path}")
            return

        ext_list = ('.jpg', '.jpeg', '.png', '.bmp')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(ext_list)]
        if not files:
            print("⚠️ 该文件夹下没有找到支持格式的图片。")
            return

        print(f"📂 开始处理文件夹: {folder_path} (类别ID: {class_id})")
        print(f"📝 共发现 {len(files)} 张图片待处理...")

        total_start = time.perf_counter()
        success_count = 0

        for i, file_name in enumerate(files):
            img_path = os.path.join(folder_path, file_name)
            yolo_str = self._image_to_yolo(img_path, class_id)

            if yolo_str:
                txt_path = os.path.splitext(img_path)[0] + ".txt"
                with open(txt_path, "w", encoding='utf-8') as f:
                    f.write(yolo_str)
                success_count += 1

            # 每 50 张打印一次进度
            if (i + 1) % 50 == 0:
                print(f"⏳ 已处理 {i+1}/{len(files)} 张...")

        total_time = time.perf_counter() - total_start
        print(f"\n✅ 处理完毕！成功自动标注 {success_count} 张叶片图片。")
        print(f"⏱️ 总耗时: {total_time:.2f} 秒")
        print(f"💡 生成的 .txt 文件已保存在原图文件夹中。")


# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == "__main__":
    # 使用 Tkinter 弹出文件夹选择对话框
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="请选择包含图片的文件夹")
    if not folder_path:
        print("未选择文件夹，程序退出。")
        sys.exit(0)

    try:
        class_id = int(input("请输入该类别的编号 (例如 Healthy 是 0, 则输入 0): ").strip())
    except ValueError:
        print("❌ 类别编号必须是整数，程序退出。")
        sys.exit(1)

    labeler = BatchYOLOLabeler()
    labeler.process_folder(folder_path, class_id)

    input("\n按回车键退出...")