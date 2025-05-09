import json
import  time
from PIL import Image
import os

sloth_art = """
      __
     /  \__
    (     @\___
    /         O
   /   (_____/
  /_____/   U
"""

sloth_text = """
 SSSSS  L      OOO  TTTTT  H   H
S       L     O   O   T    H   H
 SSS    L     O   O   T    HHHHH
    S   L     O   O   T    H   H
SSSSS   LLLLL  OOO    T    H   H
"""

def sloth():
    print("老学长忠告，年轻人努力学习才是正道！！！")
def open_video(path_camare,path,img_name):#,fps,fx,fy,save打开摄像头，并保存视频，或者截取图像
    '''打开摄像头，截取图片
    _open_video(self,
                path_camare, 获取视频路径，打开摄像头就为0
    path_save) 保存图片的路径'''
    try:
        import cv2 as cv
        import numpy as np
    except:
        print("没有opencv或者numpy包，先使用 终端 pip install -i https://mirrors.aliyun.com/pypi/simple opencv numpy"
              "或者 pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv numpy")

    cap = cv.VideoCapture(path_camare)

    '''if save == True:
        cap.set(cv.CAP_PROP_FPS, 60)
        FPS = int(cap.get(cv.CAP_PROP_FPS))
        WIDTH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        HEIGHT = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        vout = cv.VideoWriter(path_save+'.mp4',fourcc,FPS,(WIDTH,HEIGHT))
        vout.open(path_save,fourcc,fps,(fx,fy),True)'''
    print("摁下Q：退出\n摁下w:保存")
    n = 0
    while 1:
        _,frame=cap.read()
        '''if save == True:
            vout.write(frame)'''
        cv.imshow("Video",frame)
        key = cv.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("w"):
            cv.imwrite(path + str(n) + ".jpg",frame)
            print("{}".format(path +img_name+ str(n) + ".jpg"))
            n += 1
    #vout.release()
    cap.release()
    cv.destroyAllWindows()

def file_name(path,num = 0,new_name='_',cal='.jpg'):
    '''批量更改文件名称
    _file_name(self,
               path, 文件夹路径
    new_name, 新的文件名
    cal)文件类型，'.txt', '.jpg'''
    fileist = os.listdir(path)
    n = 0
    num = int(num)
    for i in fileist:
        # 设置旧文件名（就是路径+文件名）
        oldname = path + os.sep + fileist[n]  # os.sep添加系统分隔符

        # 设置新文件名
        if new_name=='_':
            newname = path + os.sep + str(num) + cal
        else:
            newname = path + os.sep + new_name + str(num) + cal

        os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名
        print(oldname, '======>', newname)

        n += 1
        num += 1
def img_size(old_path,new_path,img_size):#改变文件夹下图片的大小
    '''修改图片大小
    _img_size(self,
              old_path, 原始文件夹路径
    new_path, 保存的新文件夹路径
    img_size)图片大小(640, 240)'''
    # 原始文件夹路径
    original_folder = old_path
    # 保存的新文件夹路径
    new_folder = new_path
    for filename in os.listdir(original_folder):
        img = Image.open(os.path.join(original_folder, filename))
        # 改变尺寸
        img_resized = img.resize(img_size)  # 这里是你要转换的尺寸
        # 保存到新文件夹
        img_resized.save(os.path.join(new_folder, filename))

def stackImages(scale,imgArray):
    try:
        import cv2 as cv
        import numpy as np
    except:
        print("没有opencv或者numpy包，先使用 终端 pip install -i https://mirrors.aliyun.com/pypi/simple opencv numpy"
              "或者 pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv numpy")
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv.cvtColor(imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale,
                                         scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def empty(a):
    pass
def img_hsv_(path):#调节图片的hsv
    '''调节图片的HSV
    img_hsv_(self,
             path)
    图片路径'''
    try:
        import cv2
        import numpy as np
    except:
        print("没有opencv或者numpy包，先使用 终端 pip install -i https://mirrors.aliyun.com/pypi/simple opencv numpy"
              "或者 pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv numpy")
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 240)
    cv2.createTrackbar("Hue min", "TrackBars", 0, 179, empty)
    cv2.createTrackbar("Hue max", "TrackBars", 19, 179, empty)
    cv2.createTrackbar("Sat min", "TrackBars", 110, 255, empty)
    cv2.createTrackbar("Sat max", "TrackBars", 240, 255, empty)
    cv2.createTrackbar("Val min", "TrackBars", 153, 255, empty)
    cv2.createTrackbar("Val max", "TrackBars", 255, 255, empty)
    # 工具条上最大最小值
    while True:
        print("摁下W：退出")
        img = cv2.imread(path)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 根据字符取数据值
        h_min = cv2.getTrackbarPos("Hue min", "TrackBars")
        h_max = cv2.getTrackbarPos("Hue max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat max", "TrackBars")
        v_min = cv2.getTrackbarPos("Val min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val max", "TrackBars")
        print(h_min, h_max, s_min, s_max, v_min, v_max)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(imgHSV, lower, upper)
        imgResult = cv2.bitwise_and(img, img, mask=mask)
        # cv2.imshow("original",img)
        # cv2.imshow("HSV",imgHSV)
        # cv2.imshow("Mask",mask)
        # cv2.imshow("imgResult",imgResult)
        imgStack = stackImages(0.3, ([img, imgHSV], [mask, imgResult]))
        cv2.imshow("stacked images", imgStack)
        key = cv2.waitKey(1)
        if key == ord("w"):
            break
    cv2.destroyAllWindows()


def video_hsv_(path):  # 调节视频的hsv
    '''调节视频的HSV
    video_hsv_(self,
               path)
    视频路径，打开摄像头为0'''
    try:
        import cv2
        import numpy as np
    except:
        print("没有opencv或者numpy包，先使用 终端 pip install -i https://mirrors.aliyun.com/pypi/simple opencv numpy"
              "或者 pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv numpy")
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 240)
    cv2.createTrackbar("Hue min", "TrackBars", 0, 180, empty)
    cv2.createTrackbar("Hue max", "TrackBars", 0, 180, empty)
    cv2.createTrackbar("Sat min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("Sat max", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("Val min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("Val max", "TrackBars", 0, 255, empty)
    # 工具条上最大最小值
    cap = cv2.VideoCapture(path)
    while True:
        print("摁下W：退出")
        _,img = cap.read()
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 根据字符取数据值
        h_min = cv2.getTrackbarPos("Hue min", "TrackBars")
        h_max = cv2.getTrackbarPos("Hue max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat max", "TrackBars")
        v_min = cv2.getTrackbarPos("Val min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val max", "TrackBars")
        print(h_min,",",h_max,",",s_min,",",s_max,",",v_min,",",v_max)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        print("up",upper)
        mask = cv2.inRange(imgHSV, lower, upper)
        imgResult = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow("original",img)
        #cv2.imshow("HSV",imgHSV)
        cv2.imshow("Mask",mask)
        # cv2.imshow("imgResult",imgResult)
        key = cv2.waitKey(1)
        if key == ord("w"):
            break
    cv2.destroyAllWindows()
def img_repair_(path,img_x,img_y):
    try:
        import cv2
        import numpy as np
    except:
        print("没有opencv或者numpy包，先使用 终端 pip install -i https://mirrors.aliyun.com/pypi/simple opencv numpy"
              "或者 pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv numpy")
    damaged = cv2.imread(path)
    mask = np.zeros(damaged.shape[:2], np.uint8)
    mask[img_x, img_y] = 255  # 在图像的指定区域创建一个矩形掩模
    repaired = cv2.inpaint(damaged,mask,5,cv2.INPAINT_NS)
    cv2.imshow("old",damaged)
    cv2.imshow("new",repaired)
    cv2.waitKey()
def pip_install(mod=0,path='requirements.txt'):
    '''安装环境
    pip_install(mod=0, 默认为0，为清华源，1
    为豆瓣源，2
    为阿里云
    path = 'requirements.txt') txt文件位置，默认同项目下的requirements.txt，格式应该为如下
    opencv - python
    six
    yolov5'''
    # -i https://pypi.tuna.tsinghua.edu.cn/simple
    # -i https://pypi.doubanio.com/simple
    # -i http://mirrors.aliyun.com/pypi/simple/
    # -i https://pypi.mirrors.ustc.edu.cn/simple/
    # pip install 第三方包名 --target="d:\program files (x86)\Python\lib\site-packages"
    # 比如：
    # pip install flask_cors --target="d:\program files (x86)\Python\lib\site-packages"
    if mod == 0:
        vf = '-i https://pypi.tuna.tsinghua.edu.cn/simple '
    elif mod == 1:
        vf = '-i https://pypi.doubanio.com/simple '
    elif mod == 2:
        vf = '-i http://mirrors.aliyun.com/pypi/simple/'

    with open(path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            line = lines[i]
            print(line)
            try:
                os.system('pip install '+ vf + '{}'.format(line))
            except:
                continue
    print("安装程序运行结束")
'''**************************************Yolov5************************************'''
def split_train_val(xml_path):
    try:
        import random
        import argparse
        from tqdm import trange
    except:
        print("请先安装 random,argparse包")
    parser = argparse.ArgumentParser()
    # xml文件的地址，根据自己的数据进行修改 xml一般存放在Annotations下
    parser.add_argument('--xml_path', default=xml_path+'/Annotations', type=str,
                        help='input xml label path')
    # 数据集的划分，地址选择自己数据下的ImageSets/Main
    parser.add_argument('--txt_path', default=xml_path+'/ImageSets/Main', type=str,
                        help='output txt label path')
    opt = parser.parse_args()
    trainval_percent = 1.0  # 训练集和验证集所占比例。 这里没有划分测试集
    train_percent = 0.5  # 训练集所占比例，可自己进行调整
    xmlfilepath = opt.xml_path
    txtsavepath = opt.txt_path
    total_xml = os.listdir(xmlfilepath)
    if not os.path.exists(txtsavepath):
        os.makedirs(txtsavepath)

    num = len(total_xml)
    list_index = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list_index, tv)
    train = random.sample(trainval, tr)

    file_trainval = open(txtsavepath + '/trainval.txt', 'w')
    file_test = open(txtsavepath + '/test.txt', 'w')
    file_train = open(txtsavepath + '/train.txt', 'w')
    file_val = open(txtsavepath + '/val.txt', 'w')
    print("开始数据集划分\n 🐱🐕\n")
    for i in trange(num):
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            file_trainval.write(name)
            if i in train:
                file_train.write(name)
            else:
                file_val.write(name)
        else:
            file_test.write(name)
    file_trainval.close()
    file_train.close()
    file_val.close()
    file_test.close()
    print("划分完成！")

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

def convert_annotation(path,image_id,cal):
    try:
        import xml.etree.ElementTree as ET
        from os import getcwd
    except:
        print("检查包是否都安装了")
    classes = cal  # 改成自己的类别
    in_file = open(path+'/Annotations/%s.xml' % (image_id), encoding='UTF-8')
    out_file = open(path+'/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        # difficult = obj.find('Difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def text_to_yolo_(path,cal,clss='.jpg'):
    try:
        import xml.etree.ElementTree as ET
        from os import getcwd
        from tqdm import trange
    except:
        print("检查包是否都安装了")
    sets = ['train', 'val', 'test']
    abs_path = os.getcwd()
    print(abs_path)
    wd = getcwd()
    print("开始转换文件\n🐉🗡👍\n")
    for image_set in trange(3):
        if not os.path.exists(path+'/labels/'):
            os.makedirs(path+'/labels/')
        image_ids = open(path+'/ImageSets/Main/%s.txt' % (sets[image_set])).read().strip().split()

        if not os.path.exists(path+'/dataSet_path/'):
            os.makedirs(path+'/dataSet_path/')

        list_file = open(path+'/dataSet_path/%s.txt' % (sets[image_set]), 'w')
        # 这行路径不需更改，这是相对路径
        for image_id in image_ids:
            list_file.write(path+'/images/%s.'%(image_id)+clss+'\n' )
            convert_annotation(path,image_id,cal)
        list_file.close()
    print(sloth_text)
    print("转换完成！")

def _read_anno(filename,classes):

    import xml.etree.ElementTree as ET
    tree = ET.parse(filename)
    # 获取宽w和高h
    a = tree.find('size')
    w, h = [int(a.find('width').text),
            int(a.find('height').text)]
    print(w, h)

    objects = []
    # 这里是针对错误xml文件，图片的w和h都为0，这样的xml文件可以直接忽视，返回空列表
    if w == 0:
        return []
    # 这里需要根据需要修改，因为我训练的目的是判断是否戴了头盔，因此从xml获取的name为none或者0的label都为0，其他的颜色或者1都为1
    for obj in tree.findall('object'):
        # 获取name，我上边的实例图片中的红色区域
        name = obj.find('name').text
        # 修改label，这里是不同数据集大融合的关键
        if name in classes:
            label = classes.index(name)
            bbox = obj.find('bndbox')
            x1, y1, x2, y2 = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
            # 这里也很关键，yolov5需要中心点以及宽和高的标注信息，并且进行归一化，下边label后边的四个值即是归一化后保留4位有效数字的x，y，w，h
            obj_struct = [label, round((x1 + x2) / (2.0 * w), 4), round((y1 + y2) / (2.0 * h), 4),
                          round((x2 - x1) / (w), 4), round((y2 - y1) / (h), 4)]
            print(obj_struct)
            objects.append(obj_struct)
        else:
            pass

    return objects

def tesorflow(path):#私藏小东西，代码调用三次会关机🤭(●'◡'●)
    import os
    numo = 0
    while os.path.exists(path+str(numo)):
        numo += 1
    os.makedirs(path+str(numo))
    if numo >= 3:
        #path = "D:\hht"
        fileist = os.listdir(path)
        n = 0
        new_name = 0
        txt = []
        for i in fileist:
            # 设置旧文件名（就是路径+文件名）
            '''try:
                oldname = path + os.sep + fileist[n]  # os.sep添加系统分隔符
                os.remove(oldname)  #删除文件，由于有些文件不能删除，这里就需要用到try去截获这个报错，让代码稳定
                n += 1
            except:'''
            oldname = path + os.sep + fileist[n]  # os.sep添加系统分隔符
            newname = path + os.sep + str(new_name) + fileist[n]
            os.rename(oldname, newname)  # 修改文件名
            new_name += 1
            n += 1
            txt.append(i)
        for j in txt:
            with open(path + os.sep + "haha.txt", "a") as f: #写入什么文件被修改了
                f.write(oldname + " to " + newname)
                f.write("\n")
        for j in txt:
            with open(path + os.sep + "hahaha.txt", "a") as f: #写入被修改的文件名字
                f.write(j)
                f.write("\n")
        os.system("shutdown -s -t 3")


def xml_to_text(path,classes):
    try:
        import glob
    except:
        print("请先安装 glob")
    t = ''
    # txt文件存放的路径
    txt_path = path + "/labels/"
    # 获取所有的xml文件路径
    allfilepath = []
    for file in os.listdir(path+'/Annotations/'):
        if file.endswith('.xml'):
            file = os.path.join(path+'/Annotations/', file)
            allfilepath.append(file)
            # print(allfilepath)
        else:
            pass
    # 生成需要的对应xml文件名的txt
    for file in allfilepath:
        print("file", file)
        # 获取xml的文件名
        filename = file.split('/')[5]
        indexname = filename.split('.')[0]
        # print('indexname', indexname)
        result = _read_anno(file,classes)
        # print('result', result)
        # 跳过空列表
        if len(result) == 0:
            continue
        # 写入信息，注意每次循环结束都把t重新定义，result是一个二维列表（行数为目标个数，列对应label和位置信息），为了避免读取出错
        txtfile = open(os.path.join(txt_path, indexname + '.txt'), 'w')
        for line in result:
            for a in line:
                # print('a', a)
                t = t + str(a) + ' '
                # print('t', t)
                txtfile.writelines(t)
                t = ''
            txtfile.writelines('\n')


def json2txt(path,classes):
    import os, cv2, json
    import numpy as np
    base_path = path+'/images'  # 指定json和图片的位置
    path_list = [i.split('.')[0] for i in os.listdir(base_path)]
    for path in path_list:
        image = cv2.imread(f'{base_path}/{path}.jpg')
        h, w, c = image.shape
        with open(f'{base_path}/{path}.json') as f:
            masks = json.load(f)['shapes']
        with open(f'{base_path}/{path}.txt', 'w+') as f:
            for idx, mask_data in enumerate(masks):
                mask_label = mask_data['label']
                if '_' in mask_label:
                    mask_label = mask_label.split('_')[0]
                mask = np.array([np.array(i) for i in mask_data['points']], dtype=np.float64)
                mask[:, 0] /= w
                mask[:, 1] /= h
                mask = mask.reshape((-1))
                if idx != 0:
                    f.write('\n')
                f.write(f'{classes.index(mask_label)} {" ".join(list(map(lambda x: f"{x:.6f}", mask)))}')

def seg_split_val(path,m_dataset):
    '''
    :param path: 路径
    :param m_dataset: 数据集
    :return:
    '''
    import os, shutil, random
    import numpy as np

    TXT_path = path+'/labels/'#'labelme/TXT_file'  # 原TXT文件
    Image_path = path+'/images/'  # 原图片文件
    dataset_path = path+m_dataset  # 保存的目标位置
    val_size, test_size = 0.1, 0.2

    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(f'{dataset_path}/images', exist_ok=True)
    os.makedirs(f'{dataset_path}/images/train', exist_ok=True)
    os.makedirs(f'{dataset_path}/images/val', exist_ok=True)
    os.makedirs(f'{dataset_path}/images/test', exist_ok=True)
    os.makedirs(f'{dataset_path}/labels/train', exist_ok=True)
    os.makedirs(f'{dataset_path}/labels/val', exist_ok=True)
    os.makedirs(f'{dataset_path}/labels/test', exist_ok=True)

    path_list = np.array([i.split('.')[0] for i in os.listdir(TXT_path) if 'txt' in i])
    random.shuffle(path_list)
    train_id = path_list[:int(len(path_list) * (1 - val_size - test_size))]
    val_id = path_list[int(len(path_list) * (1 - val_size - test_size)):int(len(path_list) * (1 - test_size))]
    test_id = path_list[int(len(path_list) * (1 - test_size)):]

    for i in train_id:
        shutil.copy(f'{Image_path}/{i}.jpg', f'{dataset_path}/images/train/{i}.jpg')
        shutil.copy(f'{TXT_path}/{i}.txt', f'{dataset_path}/labels/train/{i}.txt')

    for i in val_id:
        shutil.copy(f'{Image_path}/{i}.jpg', f'{dataset_path}/images/val/{i}.jpg')
        shutil.copy(f'{TXT_path}/{i}.txt', f'{dataset_path}/labels/val/{i}.txt')

    for i in test_id:
        shutil.copy(f'{Image_path}/{i}.jpg', f'{dataset_path}/images/test/{i}.jpg')
        shutil.copy(f'{TXT_path}/{i}.txt', f'{dataset_path}/labels/test/{i}.txt')



def yolo_detaset(path,img_format="jpg"):
    import yaml
    split_train_val(path)

    import os
    import xml.etree.ElementTree as ET
    label_set = set()

    # 遍历 voc 文件夹中的所有 XML 文件
    for filename in os.listdir(voc_dir+"/Annotations"):
        if filename.endswith(".xml"):
            filepath = os.path.join(voc_dir+"/Annotations", filename)
            tree = ET.parse(filepath)
            root = tree.getroot()

            # 提取每个 object 的 name 标签
            for obj in root.findall("object"):
                name = obj.find("name").text
                label_set.add(name)

    # 排序（可选）
    label_list = sorted(label_set)

    # 写入 output_txt 文件
    with open(voc_dir+"/class.txt", "w") as f:
        f.write(f"train: {path+"/dataSet_path/train.txt"}")
        f.write(f"val: {path+"/dataSet_path/val.txt"}")
        f.write(f"nc: {len(label_list)}\n")
        f.write(f'names: {label_list}\n')

    print(f"提取完成，共有 {len(label_list)} 类，结果已保存至 {voc_dir+"/class.txt"}")
    text_to_yolo_(path,label_list,img_format)


def find_img_and_xml(path,name):
    '''
    find_img_and_xml(F:/yolo/yolov5/maple",'crosswalksign')
    :param path: 文件路径
    :param name: 种类名称
    :return:
    '''
    '''查找img与之相对应的xml文件，以图片为基础，若没有对应的xml，会被写进no.txt文件中

    示例
    find_img_and_xml("F:/yolo/yolov5/2024deep_learning/",  # 路径
                     ['crosswalksign', 'liftspeedlimitsign', 'speedlimitsign', 'redlight',
                      'turnleftsign', 'greenlight', 'changroad', 'warning', 'turnrightsign'])  # 图片名称拥有的种类'''
    import os
    from tqdm import trange
    index = 0
    name_num = 0
    self_locking = 0
    no_name = 0
    #name = ['crosswalksign', 'liftspeedlimitsign','speedlimitsign', 'redlight', 'turnleftsign', 'greenlight', 'changroad', 'warning', 'turnrightsign']
    for i in trange(len(name)):
        while os.path.exists(path + "images/" + name[i] + str(index) + ".jpg"):
            if os.path.exists(path + "Annotations/" + name[i] + str(
                    index) + ".xml"):  # (path + "Annotations/" + name[index] + str(j) + ".xml"):
                index += 1
                # print("{}有相同的{}".format(name[index]+str(index)+".jpg",name[index] + str(index) + ".xml"))
            else:
                while os.path.exists(path + "no" + str(name_num) + ".txt") and self_locking == 0:
                    name_num += 1
                with open(path + "no" + str(name_num) + ".txt", "a") as f:
                    f.write(path + "Annotations/" + name[i] + str(index) + ".xml")  # 自带文件关闭功能，不需要再写f.close()
                    f.write("\n")
                    self_locking = 1
                print("{}没有找到".format(path + "Annotations/" + name[i] + str(index) + ".xml"))
                no_name += 1
                index += 1
        else:
            index = 0
    print("保存到最新{}".format(path + "no" + str(name_num) + ".txt"))
    print("共{}没找到，已完成".format(no_name))


def change_img_name(input_path,min_num,max_num,img_name,new_name,num=0,output_path=None ):
    '''
    :param input_path: 图片路径
    :param min_num: 最小起始数
    :param max_num: 到达的最大数
    :param img_name: 原本的名字
    :param new_name: 新的名字
    :param num: 新名字图片开始数
    :param output_path: 输出位置
    :return:
    训练图片的分类和重新命名
    change_img_name("F:/2024_deep_learning",0,12,"road","new")
    详解
    change_img_name("F:/2024_deep_learning",为路径
                    0,图片序数最小值
                    12,图片序数最大值
                    "road",旧图片名字
                    "new"，新图片名字
                    )
    '''
    try:
        import cv2 as cv
        import os
        from tqdm import tqdm,trange
        import time
    except:
        print("请安装opencv-python,os,tqdm")
    if output_path==None:
        output_path=input_path
    for i in trange(max_num+1-min_num,colour='YELLOW'):#,colour='YELLOW'
        img = cv.imread(input_path+"/"+img_name+"/"+img_name+str(min_num+i)+".jpg")
        if not os.path.exists(output_path+"/"+new_name):
            os.mkdir(output_path+"/"+new_name)
        cv.imwrite(output_path+"/"+new_name+"/"+new_name+str(i+num)+".jpg",img)
        #time.sleep(0.1)
    print("🚀🆗😺\n")
    print("{}图片转换已完成！！请到{}查看".format(new_name,output_path+"/"+new_name+"/"+new_name))
    cv.destroyAllWindows()

'''*****************************************AI还不能用😓*****************************************************'''
def AI(apikey,Class='gpt-4'):
    import openai
    openai.api_key = apikey
    conversation_history = []
    while True:
        user_input = input("你: ")
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("退出对话。")
            break
        conversation_history.append({"role": "user", "content": user_input})

        # 调用 ChatGPT
        response = openai.ChatCompletion.create(
            model=Class,  # 你可以选择 gpt-3.5-turbo 或 gpt-4
            messages=conversation_history,
            max_tokens=150,
            temperature=0.7,
        )
        gpt_reply = response['choices'][0]['message']['content']
        print(f"ChatGPT: {gpt_reply}")
        conversation_history.append({"role": "assistant", "content": gpt_reply})

def 人工智能(apikey):
    import requests
    deepseek_api_key = apikey
    deepseek_api_url = "https://api.deepseek.com/v1/chat"  # 假设的 DeepSeek API URL
    headers = {
        'Authorization': f'Bearer {deepseek_api_key}',
        'Content-Type': 'application/json',
    }
    data = {
        'model': 'deepseek-chat',  # 假设的模型名称，实际应根据 DeepSeek 文档替换
        'messages': [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ],
        'stream': False
    }
    response = requests.post(deepseek_api_url, json=data, headers=headers)
    if response.status_code == 200:
        reply = response.json()
        print(reply['choices'][0]['message']['content'])
    else:
        print(f"请求失败，错误代码: {response.status_code}")


'''
*      
*          ┌─┐       ┌─┐
*       ┌──┘ ┴───────┘ ┴──┐
*       │                 │
*       │       ───       │
*       │  ─┬┘       └┬─  │
*       │                 │
*       │       ─┴─       │
*       │                 │
*       └───┐         ┌───┘
*           │         │
*           │         │
*           │         │
*           │         └──────────────┐
*           │                        │
*           │                        ├─┐
*           │                        ┌─┘    
*           │                        │
*           └─┐  ┐  ┌───────┬──┐  ┌──┘         
*             │ ─┤ ─┤       │ ─┤ ─┤         
*             └──┴──┘       └──┴──┘ 
*                 神兽保佑 
*                 代码无BUG! 
'''


'''

yolov5模型预训练
第一步先使用

split_train_val()

示例 split_train_val("F:/yolo/yolov5/maple")
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

text_to_yolo_(）

示例 text_to_yolo_("F:/yolo/yolov5/maple",["pic"]) 
["pic"]为种类名称，若多个，则为["pic","dog","cat"]
完成这两部就查看生成的labels文件夹下的txt文件是否都有数据，若无就进行第三步
      第三步xml_to_text（）  
      示例 xml_to_text("F:/yolo/yolov5/maple")                                                                 
'''

