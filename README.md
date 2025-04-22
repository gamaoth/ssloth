# ssloth
只是一个提供给实验室入门的包
# yolov5模型预训练
## 对于标注出的文件是.xml后缀的
第一步先使用

split_train_val()

示例 

	split_train_val("F:/yolo/yolov5/maple")
 
则这个文件夹下应该包含存放xml后缀的文件夹命名为Annotations,
jpg后缀的文件夹命名为images,

示意图如下
                             
    maple                                        
        |
        |---Annotations
        |   |---.xml文件
        |
        |---images
            |---.jpg文件                                                                                                                             
                                                                               
第二步

text_to_yolo_()
示例 

    text_to_yolo_("F:/yolo/yolov5/maple",["pic"]) 
		
["pic"]为种类名称，若多个，则为["pic","dog","cat"]

完成这两部就查看生成的labels文件夹下的txt文件是否都有数据，若无就进行第三步
      
第三步xml_to_text（）  
示例 

    xml_to_text("F:/yolo/yolov5/maple")                                                                 
## 对于标注出的文件是.txt后缀的
待开发
## 寻找images和Annotations里面的文件是否一一对应

    find_img_and_xml(F:/yolo/yolov5/maple",'crosswalksign')
    :param path: 文件路径
    :param name: 种类名称
    :return:

查找img与之相对应的xml文件，以图片为基础，若没有对应的xml，会被写进no.txt文件中
示例

    find_img_and_xml("F:/yolo/yolov5/2024deep_learning/",  # 路径
                     ['crosswalksign', 'liftspeedlimitsign', 'speedlimitsign', 'redlight',
                      'turnleftsign', 'greenlight', 'changroad', 'warning', 'turnrightsign']) 
		      
图片名称拥有的种类
## 打开摄像头拍照
打开摄像头，截取图片
open_video(path_camare, 获取视频路径，打开摄像头就为0
    path_save) 保存图片的路径
示例

	open_video(0,"F:/sloth/","sloth")
 
