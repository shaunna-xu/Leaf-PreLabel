import cv2
import numpy as np
from cv_utils import grabCut_adaptive


class LeafDetector:
    """
    叶片检测器，支持三种分割模式：
    1. 'seed'     ：OTSU 二值化找质心 → 构建种子掩膜 → 单次自适应 GrabCut
    2. 'padding'  ：边缘复制（2像素）→ 矩形初始化 → 单次自适应 GrabCut
    3. 'twostage' ：粗分割（矩形留边5%）找质心 → 细分割（掩膜初始化）→ 两阶段自适应 GrabCut
    """

    # GrabCut GMM 模型（类属性，所有实例共享）
    BGD = np.zeros((1, 65), np.float64)
    FGD = np.zeros((1, 65), np.float64)

    def __init__(self, mode='seed'):
        """
        参数:
            mode (str): 分割模式，可选 'seed'、'padding'、'twostage'，默认 'seed'。
        """
        self.mode = mode
        self.mask = None

    def detect(self, img):
        """
        检测叶片区域，返回二值掩膜。

        参数:
            img (np.ndarray): BGR 图像（任意尺寸）。

        返回:
            np.ndarray: 二值掩膜（255=叶片, 0=背景），与输入图像同尺寸。
        """
        if self.mode == 'seed':
            return self._detect_seed(img)
        elif self.mode == 'padding':
            return self._detect_padding(img)
        elif self.mode == 'twostage':
            return self._detect_twostage(img)
        else:
            raise ValueError(f"未知模式: {self.mode}，请选择 'seed'、'padding' 或 'twostage'")

    # --------------------------------------------------------------------------
    # 模式1：种子点引导
    # --------------------------------------------------------------------------
    def _detect_seed(self, img):
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # OTSU 二值化（反转，使叶片为白色）
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((15, 15), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        body = self._largest_component(thresh)
        if body is None:
            cx, cy = w // 2, h // 2
        else:
            M = cv2.moments(body)
            if M['m00'] < w * h * 0.10:
                cx, cy = w // 2, h // 2
            else:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

        # 构建掩膜
        mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        max_edge = max(h, w)
        seed_r = max(15, int(max_edge * 0.04))
        prob_r = max(60, int(max_edge * 0.13))

        mask[max(0, cy-seed_r):min(h, cy+seed_r),
            max(0, cx-seed_r):min(w, cx+seed_r)] = cv2.GC_FGD
        mask[max(0, cy-prob_r):min(h, cy+prob_r),
            max(0, cx-prob_r):min(w, cx+prob_r)] = cv2.GC_PR_FGD

        # # 边缘背景,如果叶片没有紧贴图像边界，可以定义边缘为确定背景
        # mask[0, :] = cv2.GC_BGD
        # mask[-1, :] = cv2.GC_BGD
        # mask[:, 0] = cv2.GC_BGD
        # mask[:, -1] = cv2.GC_BGD

        # ★★★ 核心：捕获 GMM 初始化失败，回退到颜色分割 ★★★
        try:
            _ = grabCut_adaptive(img, mask, None, self.BGD, self.FGD,
                                max_iters=10, tol=0.005, mode=cv2.GC_INIT_WITH_MASK)
        except cv2.error as e:
            if "initGMMs" in str(e):
                print("⚠️ 前景/背景颜色不可分，回退到颜色分割 (seed -> fallback)")
                # 使用颜色回退方法
                leaf = self._color_fallback(img)
                if leaf is None:
                    leaf = np.zeros((h, w), dtype=np.uint8)
                leaf = self._postprocess(leaf, h, w)
                self.mask = leaf
                return leaf
            else:
                raise  # 其他 cv2 错误，重新抛出

        leaf = self._largest_component(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
        )
        if leaf is None:
            leaf = np.zeros((h, w), dtype=np.uint8)

        leaf = self._postprocess(leaf, h, w)
        self.mask = leaf
        return leaf

    # --------------------------------------------------------------------------
    # 模式2：边缘复制 Padding
    # --------------------------------------------------------------------------
    def _detect_padding(self, img):
        h, w = img.shape[:2]
        pad = 10
        img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        h_pad, w_pad = img_pad.shape[:2]
        rect = (pad, pad, w, h)

        mask_pad = np.zeros((h_pad, w_pad), dtype=np.uint8)
        _ = grabCut_adaptive(img_pad, mask_pad, rect, self.BGD, self.FGD,
                             max_iters=10, tol=0.005, mode=cv2.GC_INIT_WITH_RECT)

        mask_cropped = mask_pad[pad:-pad, pad:-pad]
        leaf = self._largest_component(
            (mask_cropped == cv2.GC_FGD) | (mask_cropped == cv2.GC_PR_FGD)
        )
        if leaf is None:
            leaf = np.zeros((h, w), dtype=np.uint8)

        leaf = self._postprocess(leaf, h, w)
        self.mask = leaf
        return leaf

    # --------------------------------------------------------------------------
    # 模式3：两阶段分割
    # --------------------------------------------------------------------------
    def _detect_twostage(self, img):
        h, w = img.shape[:2]

        # ---- 第一阶段：粗分割（矩形留边5%，固定1次迭代） ----
        rect = (int(w * 0.05), int(h * 0.05), int(w * 0.90), int(h * 0.90))
        gc1 = np.zeros((h, w), np.uint8)
        cv2.grabCut(img, gc1, rect, self.BGD, self.FGD, 1, cv2.GC_INIT_WITH_RECT)

        body = self._largest_component((gc1 == cv2.GC_FGD) | (gc1 == cv2.GC_PR_FGD))
        if body is None:
            body = self._color_fallback(img)
            if body is None:
                return np.zeros((h, w), dtype=np.uint8)

        # ---- 计算质心 ----
        M = cv2.moments(body)
        if M['m00'] == 0:
            self.mask = body
            return body

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # ---- 第二阶段：细分割（掩膜初始化） ----
        gc2 = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

        gc2[body > 0] = cv2.GC_PR_FGD
        max_edge = max(h, w)
        seed_r = max(15, int(max_edge * 0.04))

        gc2[max(0, cy-seed_r):min(h, cy+seed_r),
            max(0, cx-seed_r):min(w, cx+seed_r)] = cv2.GC_FGD
        
        
        _ = grabCut_adaptive(img, gc2, None, self.BGD, self.FGD,
                             max_iters=10, tol=0.005, mode=cv2.GC_INIT_WITH_MASK)

        leaf = self._largest_component(
            (gc2 == cv2.GC_FGD) | (gc2 == cv2.GC_PR_FGD)
        )
        if leaf is None or cv2.countNonZero(leaf) < h * w * 0.02:
            leaf = body

        leaf = self._postprocess(leaf, h, w)
        self.mask = leaf
        return leaf

    # --------------------------------------------------------------------------
    # 内部辅助方法（保持不变）
    # --------------------------------------------------------------------------

    def _largest_component(self, binary):
        """提取二值图像中面积最大的连通域"""
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), 8)
        if num <= 1:
            return None
        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        out = np.zeros_like(binary, dtype=np.uint8)
        out[labels == idx] = 255
        return out

    def _color_fallback(self, img):
        """HSV 绿色范围提取作为 GrabCut 失败的回退"""
        h, w = img.shape[:2]
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        mask = cv2.inRange(h_ch, 15, 90) & cv2.threshold(s_ch, 30, 255,
                                                          cv2.THRESH_BINARY)[1]

        scale = min(h, w) / 1000.0
        k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (max(3, int(3*scale)|1),) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_small)
        k_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (max(5, int(11*scale)|1),) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_large)
        comp = self._largest_component(mask)
        return comp if comp is not None else mask

    def _postprocess(self, mask, h, w):
        """简单形态学后处理（闭运算 + 开运算）"""
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