# -*- coding: utf-8 -*-
import uuid

import requests

# url = 'http://127.0.0.1:5000/uploadImage'
url = 'http://47.93.15.79:5000/uploadImage'

# files = {'file': open("/usr/local/spider/image/11.bmp", 'rb')}
# files = {'file': open("d:/work/captcha/panjueshu/12.bmp", 'rb')}
# res = requests.post(url, files=files)
# print(res)
# uuid.uuid1()+'_'+'0/1'+'_'+'value.bmp'
url = 'http://47.93.15.79:5000/judge'
data = {'imageName': '12.bmp', 'imageValue': 'testLog12', 'right': '0'}
res = requests.post(url=url, data=data)
print(res.content)
# import logging.handlers
#
# import time
#
# formatDate = time.strftime('%Y-%m-%d', time.localtime(time.time()))
# LOG_FILE = formatDate + 'CAPTCHA_PJS.log'
#
# handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=5)  # 实例化handler
# fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
#
# formatter = logging.Formatter(fmt)  # 实例化formatter
# handler.setFormatter(formatter)  # 为handler添加formatter
#
# logger = logging.getLogger('tst')  # 获取名为tst的logger
# logger.addHandler(handler)  # 为logger添加handler
# logger.setLevel(logging.INFO)
#
# logger.info("wwo lai ceshi ceshi ")
# logger.info("1:%s 2:%s 3:%s", '1', '2', '3')
# print(uuid.uuid1())
# # print(uuid.uuid3())
# print(uuid.uuid4())
# # print(uuid.uuid5())
