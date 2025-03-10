
import cv2
import os
from multiprocessing import Pool

img_path = '/xxx/images'
label_path = 'xxx/labelTxt/'
new_img_path='xxx/images'
new_label_path='xxx/labelTxt'

if not os.path.exists(new_img_path):
    print('create img path')
    os.makedirs(new_img_path)
if not os.path.exists(new_label_path):
    print('create label path')
    os.makedirs(new_label_path)

scales=[2,4,8,16]

def process_file(args):
    filename, scale, end_name = args
    img = cv2.imread(os.path.join(img_path,filename))
    height, width = img.shape[:2]

    # 缩小图片的大小（降采样）
    dst_width = int(width / scale)
    dst_height = int(height / scale)

    if dst_height<1024 or dst_width<1024:
        print('continue --',filename)
        return 0

    filename_img=filename.split('.')[0]+end_name+'.'+filename.split('.')[1]
    filename_txt=filename.split('.')[0]+end_name+'.txt'

    img_out = cv2.resize(img, (dst_width, dst_height))
    cv2.imwrite(os.path.join(new_img_path,filename_img),img_out)

    labelpath = label_path + filename.split('.')[0] + '.txt'
    out_label_path=os.path.join(new_label_path,filename_txt)

    with open(labelpath, 'r') as f_in:  # 打开txt文件
        lines = f_in.readlines()
        splitlines = [x.strip().split(' ') for x in lines]  # 根据空格分割

        with open(out_label_path, 'w') as f_out:
            for i, splitline in enumerate(splitlines):
                x0 = float(splitline[0])/ scale
                y0 = float(splitline[1])/ scale
                x1 = float(splitline[2])/ scale
                y1 = float(splitline[3])/ scale
                x2 = float(splitline[4])/ scale
                y2 = float(splitline[5])/ scale
                x3 = float(splitline[6])/ scale
                y3 = float(splitline[7])/ scale
                class_name=splitline[8]
                difficult=splitline[9]

                f_out.write(str(x0) + ' ' + str(y0) + ' ' + str(x1) + ' ' + str(y1) + ' '
                        + str(x2) + ' ' + str(y2) + ' ' + str(x3) + ' ' + str(y3) + ' ' + class_name + ' ' + difficult + '\n')
    return 1

for scale in scales:
    print('----下采样倍率----:',scale)
    end_name='_down'+str(scale)  # 下采样名称

    filenames = os.listdir(img_path)  # 获取每一个txt的名称

    # 使用 multiprocessing 的 Pool
    with Pool() as p:
        count = sum(p.map(process_file, [(filename, scale, end_name) for filename in filenames]))
    print('----下采样倍率----:',scale)
    print('转化数量:',count)
    print('done!')
