# ssloth
只是一个提供给实验室入门的包
# yolov5模型预训练
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
