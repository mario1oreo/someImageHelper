# coding=utf-8
# 对图片进行一些灰度化  二值化相关的处理
import stat

from PIL import Image
import base64
f=open(r'D:\work\captcha\panjueshu\3.bmp','rb') #二进制方式打开图文件
ls_f=base64.b64encode(f.read()) #读取文件内容，转换为base64编码
f.close()
print(ls_f)

def initTable():
    table = []
    for i in range(256):
        if i < 180:
            table.append(0)
        else:
            table.append(1)
    return table

stat.S_IREAD+stat.S_IWOTH
# for i in range(1, 2000):
#     im = Image.open('C:/image/panjueshu/originalImage/' + str(i) + '.bmp')
#     im = im.convert('L')
#     binaryImage = im.point(initTable(), '1')
#     region = (1,1,63,21)
#     img = binaryImage.crop(region)
#     # binaryImage.show()
#     img.save('C:/image/panjueshu/originalImageTemp/' + str(i) + '.bmp')




