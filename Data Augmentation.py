import os
import cv2
import numpy as np
from pathlib import Path
import random
import shutil
from tqdm import tqdm

class YOLODataAugmenter:
    def __init__(self, input_dir, output_dir, augmentations=None):
        """
        初始化图像数据增强器。
        
        Args:
            input_dir (str): 包含图像的输入目录
            output_dir (str): 保存增强图像的输出目录
            augmentations (dict): 数据增强参数字典
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认增强参数
        self.augmentations = augmentations or {
            'flip_horizontal': 0.5,  # 水平翻转概率
            'flip_vertical': 0.3,    # 垂直翻转概率
            'rotation': 1,          # 最大旋转角度（度）
            'brightness': 0.3,      # 最大亮度调整
            'scale': 0.1,           # 最大缩放比例
            'translate': 0.02       # 最大平移比例
        }

    def horizontal_flip(self, image):
        """水平翻转图像"""
        return cv2.flip(image, 1)

    def vertical_flip(self, image):
        """垂直翻转图像"""
        return cv2.flip(image, 0)

    def rotate(self, image, angle):
        """旋转图像，角度为 angle 度"""
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    def adjust_brightness(self, image, factor):
        """调整图像亮度"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def scale(self, image, scale_factor):
        """缩放图像"""
        h, w = image.shape[:2]
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        return cv2.resize(image, (new_w, new_h))

    def translate(self, image, tx, ty):
        """平移图像"""
        h, w = image.shape[:2]
        M = np.float32([[1, 0, tx*w], [0, 1, ty*h]])
        return cv2.warpAffine(image, M, (w, h))

    def augment(self, image_path, num_augmentations=3):
        """应用数据增强"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Failed to load image {image_path}")
            return
        
        base_name = image_path.stem
        for i in range(num_augmentations):
            aug_image = image.copy()

            if random.random() < self.augmentations['flip_horizontal']:
                aug_image = self.horizontal_flip(aug_image)
            if random.random() < self.augmentations['flip_vertical']:
                aug_image = self.vertical_flip(aug_image)
            if self.augmentations['rotation'] > 0:
                angle = random.uniform(-self.augmentations['rotation'], self.augmentations['rotation'])
                aug_image = self.rotate(aug_image, angle)
            if self.augmentations['brightness'] > 0:
                factor = random.uniform(1 - self.augmentations['brightness'], 1 + self.augmentations['brightness'])
                aug_image = self.adjust_brightness(aug_image, factor)
            if self.augmentations['scale'] > 0:
                scale_factor = random.uniform(1 - self.augmentations['scale'], 1 + self.augmentations['scale'])
                aug_image = self.scale(aug_image, scale_factor)
            if self.augmentations['translate'] > 0:
                tx = random.uniform(-self.augmentations['translate'], self.augmentations['translate'])
                ty = random.uniform(-self.augmentations['translate'], self.augmentations['translate'])
                aug_image = self.translate(aug_image, tx, ty)

            aug_image_path = self.output_dir / f"{base_name}_aug{i}.jpg"
            cv2.imwrite(str(aug_image_path), aug_image)

    def process_dataset(self, num_augmentations=3):
        """处理输入目录中的所有图像，带进度条"""
        image_paths = list(self.input_dir.glob("*.jpg"))
        for image_path in tqdm(image_paths, desc="处理图像"):
            self.augment(image_path, num_augmentations)



if __name__ == "__main__":
    input_dir = "E:/my_yolo/my_yolo/video/test"
    output_dir = "E:/my_yolo/my_yolo/video/augmented_labels"

    augmenter = YOLODataAugmenter(input_dir, output_dir)
    augmenter.process_dataset(num_augmentations=10)
    print("Data augmentation completed!")