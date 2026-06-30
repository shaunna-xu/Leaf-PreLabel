# cv_utils.py
import cv2
import numpy as np

def grabCut_adaptive(img, mask, rect, bgdModel, fgdModel, max_iters=50,
                     tol=0.005, mode=cv2.GC_INIT_WITH_MASK):
    """
    自适应迭代的 GrabCut，当 mask 变化小于 tol 时提前停止。
    
    参数：
        img, mask, rect, bgdModel, fgdModel, mode :
            与 cv2.grabCut 完全相同。
        max_iters : 最大迭代次数（防止无限循环）
        tol : 收敛阈值（mask 变化像素数 / 总像素数）
    
    返回：
        mask : 最终的标签掩膜（原地修改）
        iters : 实际执行的迭代次数
    """
    iters = 0
    if mode != cv2.GC_INIT_WITH_MASK:
        # 若不是用掩膜初始化，仍需先执行一次初始化迭代
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 1, mode)
        mode = cv2.GC_INIT_WITH_MASK   # 后续都用 MASK 模式
        iters += 1

    h, w = mask.shape
    total_pixels = h * w
      

    for i in range(max_iters):
        old = mask.copy()
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
        iters += 1
        # 计算变化像素比例
        diff = np.sum(old != mask) / total_pixels
        if diff < tol:
            break

    return iters