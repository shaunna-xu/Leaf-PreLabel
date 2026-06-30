import os
import time
import cv2
from leaf_detector import LeafDetector

# =============================================================================
# 批量生成 YOLO 标签
# =============================================================================
class BatchYOLOLabeler:
    """批量自动标注类，利用 LeafDetector 进行分割。"""

    def __init__(self, mode='seed'):
        """
        参数:
            mode (str): 分割模式，传递给 LeafDetector，可选 'seed', 'padding', 'twostage'
        """
        self.detector = LeafDetector(mode=mode)

    def _image_to_yolo(self, img_path, class_id):
        """对单张图片生成 YOLO 标签字符串（归一化坐标）。"""
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            return None
        h_img, w_img = img.shape[:2]

        # 预处理：若图片大于320,执行下采样
        max_size = 320
        if max(h_img, w_img) > max_size:
            scale = max_size / max(h_img, w_img)
            img = cv2.resize(img, (int(w_img*scale), int(h_img*scale)),
                             interpolation=cv2.INTER_AREA)
            h_img, w_img = img.shape[:2]

        # 调用分割器（返回掩膜）
        mask = self.detector.detect(img)

        # 从掩膜提取边界框
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        max_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_cnt)

        # 过滤过小框
        if w < 10 or h < 10:
            return None

        # YOLO 格式：class_id x_center y_center width height (归一化)
        x_center = (x + w / 2.0) / w_img
        y_center = (y + h / 2.0) / h_img
        width_norm = w / w_img
        height_norm = h / h_img
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"

    def process_folder(self, folder_path, class_id, output_dir=None):
        """批量处理文件夹中的所有图片。
    
        参数:
        folder_path (str): 图片所在文件夹路径
        class_id (int): YOLO 类别编号
        output_dir (str, optional): 标签输出根目录。若提供，则在其中创建与输入文件夹同名的子目录保存 .txt；
                                    若不提供，则 .txt 保存在图片同目录下。
        """
        if not os.path.exists(folder_path):
            print(f"❌ 文件夹路径不存在: {folder_path}")
            return

        ext_list = ('.jpg', '.jpeg', '.png', '.bmp')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(ext_list)]
        if not files:
            print("⚠️ 该文件夹下没有找到支持格式的图片。")
            return

        # ---- 确定输出目录 ----
        if output_dir:
            # 获取输入文件夹的名称（例如 "Healthy"）
            folder_name = os.path.basename(folder_path.rstrip(os.sep))
            # 拼接到输出根目录下，例如 outputs/labels/Healthy
            target_dir = os.path.join(output_dir, folder_name)
            os.makedirs(target_dir, exist_ok=True)
            print(f"📁 标签将统一保存到: {target_dir}")
        else:
            target_dir = folder_path  # 默认保存在原图文件夹
            print(f"📁 标签将保存在原图文件夹中")

        print(f"📂 开始处理文件夹: {folder_path} (类别ID: {class_id})")
        print(f"📝 共发现 {len(files)} 张图片待处理...")

        total_start = time.perf_counter()
        success_count = 0

        for i, file_name in enumerate(files):
            img_path = os.path.join(folder_path, file_name)
            yolo_str = self._image_to_yolo(img_path, class_id)

            if yolo_str:
                # ---- 生成 .txt 文件路径（根据 target_dir） ----
                base_name = os.path.basename(img_path)          # 例如 "leaf_01.jpg"
                name_without_ext = os.path.splitext(base_name)[0]  # 例如 "leaf_01"
                txt_path = os.path.join(target_dir, name_without_ext + ".txt")
            
                with open(txt_path, "w", encoding='utf-8') as f:
                    f.write(yolo_str)
                success_count += 1

            if (i + 1) % 50 == 0:
                print(f"⏳ 已处理 {i+1}/{len(files)} 张...")

        total_time = time.perf_counter() - total_start
        print(f"\n✅ 处理完毕！成功自动标注 {success_count} 张叶片图片。")
        print(f"⏱️ 总耗时: {total_time:.2f} 秒")
        print(f"💡 生成的 .txt 文件已保存在: {target_dir}")