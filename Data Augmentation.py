import os
import cv2
import numpy as np
from pathlib import Path
import random
import shutil

class YOLODataAugmenter:
    def __init__(self, input_dir, output_dir, augmentations=None):
        """
        Initialize the YOLO data augmenter.
        
        Args:
            input_dir (str): Directory containing images and YOLO .txt annotations
            output_dir (str): Directory to save augmented images and annotations
            augmentations (dict): Dictionary of augmentation parameters
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default augmentation parameters
        self.augmentations = augmentations or {
            'flip_horizontal': 0.5,  # Probability of horizontal flip
            'flip_vertical': 0.3,    # Probability of vertical flip
            'rotation': 15,         # Max rotation angle in degrees
            'brightness': 0.3,      # Max brightness adjustment
            'scale': 0.2,           # Max scaling factor
            'translate': 0.1        # Max translation factor
        }

    def read_yolo_annotation(self, txt_path):
        """Read YOLO annotation file."""
        boxes = []
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    boxes.append([class_id, x_center, y_center, width, height])
        return np.array(boxes)

    def write_yolo_annotation(self, txt_path, boxes):
        """Write YOLO annotation file."""
        with open(txt_path, 'w') as f:
            for box in boxes:
                f.write(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

    def horizontal_flip(self, image, boxes):
        """Apply horizontal flip to image and boxes."""
        image = cv2.flip(image, 1)
        if len(boxes) > 0:
            boxes[:, 1] = 1 - boxes[:, 1]  # Flip x_center
        return image, boxes

    def vertical_flip(self, image, boxes):
        """Apply vertical flip to image and boxes."""
        image = cv2.flip(image, 0)
        if len(boxes) > 0:
            boxes[:, 2] = 1 - boxes[:, 2]  # Flip y_center
        return image, boxes

    def rotate(self, image, boxes, angle):
        """旋转图像和边界框，角度为 angle 度"""
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))

        if len(boxes) > 0:
            # 调试：打印 boxes 形状和内容
            print(f"Rotating image with {len(boxes)} boxes: {boxes}")

            # 将 YOLO 格式（中心点 x, y, 宽, 高）转换为四个角点
            n_boxes = boxes.shape[0]
            points = np.zeros((n_boxes, 2, 4))  # 形状：(n_boxes, 2, 4)
            points[:, 0, 0] = boxes[:, 1] - boxes[:, 3] / 2  # 左上角 x
            points[:, 1, 0] = boxes[:, 2] - boxes[:, 4] / 2  # 左上角 y
            points[:, 0, 1] = boxes[:, 1] + boxes[:, 3] / 2  # 右上角 x
            points[:, 1, 1] = boxes[:, 2] - boxes[:, 4] / 2  # 右上角 y
            points[:, 0, 2] = boxes[:, 1] - boxes[:, 3] / 2  # 左下角 x
            points[:, 1, 2] = boxes[:, 2] + boxes[:, 4] / 2  # 左下角 y
            points[:, 0, 3] = boxes[:, 1] + boxes[:, 3] / 2  # 右下角 x
            points[:, 1, 3] = boxes[:, 2] + boxes[:, 4] / 2  # 右下角 y

            # 调试：打印 points 形状
            print(f"points shape: {points.shape}")

            # 对所有角点应用旋转变换
            points = np.einsum('ij,kjl->kil', M[:, :2], points)  # 形状：(n_boxes, 2, 4)
            points += M[:, 2].reshape(1, 2, 1)  # 添加平移向量，形状：(1, 2, 1)

            # 计算旋转后边界框的最小和最大值
            x_min = points[:, 0, :].min(axis=1)  # 所有角点的 x 最小值
            x_max = points[:, 0, :].max(axis=1)  # 所有角点的 x 最大值
            y_min = points[:, 1, :].min(axis=1)  # 所有角点的 y 最小值
            y_max = points[:, 1, :].max(axis=1)  # 所有角点的 y 最大值

            # 调试：打印 x_min, x_max 等形状
            print(f"x_min shape: {x_min.shape}, x_max shape: {x_max.shape}")
            print(f"boxes[:, 1] shape: {boxes[:, 1].shape}")

            # 更新 YOLO 格式的边界框
            boxes[:, 1] = (x_min + x_max) / 2 / w  # 新 x_center
            boxes[:, 2] = (y_min + y_max) / 2 / h  # 新 y_center
            boxes[:, 3] = (x_max - x_min) / w      # 新宽度
            boxes[:, 4] = (y_max - y_min) / h      # 新高度

        return image, boxes

    def adjust_brightness(self, image, factor):
        """Adjust image brightness."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def scale(self, image, boxes, scale_factor):
        """Scale image and boxes."""
        h, w = image.shape[:2]
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        image = cv2.resize(image, (new_w, new_h))
        
        if len(boxes) > 0:
            boxes[:, [1, 3]] *= scale_factor  # Scale x_center and width
            boxes[:, [2, 4]] *= scale_factor  # Scale y_center and height
        
        return image, boxes

    def translate(self, image, boxes, tx, ty):
        """Translate image and boxes."""
        h, w = image.shape[:2]
        M = np.float32([[1, 0, tx*w], [0, 1, ty*h]])
        image = cv2.warpAffine(image, M, (w, h))
        
        if len(boxes) > 0:
            boxes[:, 1] += tx  # Translate x_center
            boxes[:, 2] += ty  # Translate y_center
        
        return image, boxes

    def augment(self, image_path, txt_path, num_augmentations=3):
        """Apply augmentations to image and annotations."""
        image = cv2.imread(str(image_path))
        boxes = self.read_yolo_annotation(txt_path)
        
        base_name = image_path.stem
        for i in range(num_augmentations):
            aug_image = image.copy()
            aug_boxes = boxes.copy() if len(boxes) > 0 else []

            # Apply augmentations with probabilities
            if random.random() < self.augmentations['flip_horizontal']:
                aug_image, aug_boxes = self.horizontal_flip(aug_image, aug_boxes)
            
            if random.random() < self.augmentations['flip_vertical']:
                aug_image, aug_boxes = self.vertical_flip(aug_image, aug_boxes)
            
            if self.augmentations['rotation'] > 0:
                angle = random.uniform(-self.augmentations['rotation'], self.augmentations['rotation'])
                aug_image, aug_boxes = self.rotate(aug_image, aug_boxes, angle)
            
            if self.augmentations['brightness'] > 0:
                factor = random.uniform(1 - self.augmentations['brightness'], 1 + self.augmentations['brightness'])
                aug_image = self.adjust_brightness(aug_image, factor)
            
            if self.augmentations['scale'] > 0:
                scale_factor = random.uniform(1 - self.augmentations['scale'], 1 + self.augmentations['scale'])
                aug_image, aug_boxes = self.scale(aug_image, aug_boxes, scale_factor)
            
            if self.augmentations['translate'] > 0:
                tx = random.uniform(-self.augmentations['translate'], self.augmentations['translate'])
                ty = random.uniform(-self.augmentations['translate'], self.augmentations['translate'])
                aug_image, aug_boxes = self.translate(aug_image, aug_boxes, tx, ty)

            # Save augmented image and annotations
            aug_image_path = self.output_dir / f"{base_name}_aug{i}.jpg"
            aug_txt_path = self.output_dir / f"{base_name}_aug{i}.txt"
            
            cv2.imwrite(str(aug_image_path), aug_image)
            if len(aug_boxes) > 0:
                self.write_yolo_annotation(aug_txt_path, aug_boxes)

    def process_dataset(self, num_augmentations=3):
        """Process all images in the input directory."""
        for image_path in self.input_dir.glob("*.jpg"):
            txt_path = self.input_dir / f"{image_path.stem}.txt"
            if txt_path.exists():
                self.augment(image_path, txt_path, num_augmentations)
            else:
                # Copy original files if no annotations
                shutil.copy(image_path, self.output_dir / image_path.name)

if __name__ == "__main__":
    input_dir = "E:/sloth"
    output_dir = "E:/sloth"

    augmenter = YOLODataAugmenter(input_dir, output_dir)
    augmenter.process_dataset(num_augmentations=10)
    print("Data augmentation completed!")