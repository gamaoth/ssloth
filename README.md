# ssloth

## 🦥 Sloth — YOLOv5 数据集预处理工具
Sloth 是一个轻量级的工具包，主要用于帮助实验室新手快速完成 YOLOv5 模型的预训练准备工作，尤其适用于从 XML 格式标注转换为 YOLO 格式的流程。

## 🧠 使用说明
### ✅ 一步完成：快速生成训练所需文件
```
yolo_detaset(path, img_format="jpg")
```
path：数据集路径

img_format：图像格式（默认 "jpg"）

示例：
```
yolo_detaset("F:/yolo/yolov5/maple")
 ```
此函数将完成数据划分、标签转换及 calss.txt 自动生成，修改成class.yaml,可直接用于 YOLOv5 训练。

### 🧩 分步骤执行（推荐了解底层过程，使用了一步完成就不需要使用分步了）

#### 第一步：划分训练集与验证集
```
split_train_val("F:/yolo/yolov5/maple")
```
数据目录结构应如下：
```
maple/
--------├── Annotations/    # 存放 XML 标注文件
--------├── images/         # 存放图像文件（.jpg）
```
#### 第二步：生成 YOLO 标签格式
```
text_to_yolo_("F:/yolo/yolov5/maple", ["pic"])
```
如有多个类别：
```
text_to_yolo_("F:/yolo/yolov5/maple", ["pic", "dog", "cat"])
```
执行完后，请检查 labels/ 文件夹下是否所有 .txt 文件都非空。

#### 第三步（如有必要）：XML 转文本
若标签文件为空或缺失，请执行：
```
xml_to_text("F:/yolo/yolov5/maple")
```
## 🔍 检查数据一致性
确保 images/ 与 Annotations/ 中的文件一一对应：
```
find_img_and_xml("F:/yolo/yolov5/2024deep_learning/",
		 ['crosswalksign', 'liftspeedlimitsign', 'speedlimitsign', 'redlight',
		  'turnleftsign', 'greenlight', 'changroad', 'warning', 'turnrightsign'])
```		   
若某图像无对应标注，将记录在 no.txt 中。

## 🎥 打开摄像头拍照
从摄像头或视频流中截取图像保存：
```
open_video(0, "F:/sloth/", "sloth")
```
0 表示使用本地摄像头

- "F:/sloth/" 为保存路径

- "sloth" 为保存文件名前缀





 
