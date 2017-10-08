#!flask/bin/python
from susongwuyoutemp import susongwuyou
# from susongwuyouBJ import susongwuyoubj
from external.org_pocoo_werkzeug.werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
import shutil,uuid
import logging.handlers
import time

formatDate = time.strftime('%Y-%m-%d', time.localtime(time.time()))
LOG_FILE = '/usr/local/spider/logCaptche/CAPTCHA_PJS_'+formatDate+'.log'

handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=5)  # 实例化handler
fmt = '%(asctime)s - %(filename)s: %(lineno)s - %(name)s - %(message)s'

formatter = logging.Formatter(fmt)  # 实例化formatter
handler.setFormatter(formatter)  # 为handler添加formatter

logger = logging.getLogger('诉讼无忧验证码')  # 获取名为tst的logger
logger.addHandler(handler)  # 为logger添加handler
logger.setLevel(logging.INFO)




app = Flask(__name__)
UPLOAD_FOLDER = '/usr/local/spider/image/right/'
UPLOAD_ERROR_FOLDER = '/usr/local/spider/image/error/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'bmp'])
IMG_TY_SUSONGWUYOU = 'susongwuyou'
IMG_TY_SUSONGWUYOU_BJ = 'susongwuyouBJ'
IMG_TY_PANJUESHU = 'panjueshu'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_ERROR_FOLDER']=UPLOAD_ERROR_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/uploadImage', methods=['GET', 'POST'])
def upload_file():
    result = {}
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            imageType = request.form.get('imageType',IMG_TY_SUSONGWUYOU)
            originalPath = UPLOAD_FOLDER + imageType + '/' + filename
            #logger.info("originalPaht:"+originalPath)
            file.save(originalPath)
            cont = ''
            if imageType == IMG_TY_SUSONGWUYOU:
                cont = susongwuyou.crack_captcha(originalPath)
                # cont=''
            elif imageType == IMG_TY_PANJUESHU:
                cont = ''#todo
            elif imageType == IMG_TY_SUSONGWUYOU_BJ:
                cont=''
                # cont = susongwuyoubj.crack_captcha(originalPath)
            imageName = str(uuid.uuid1()) + '_1_' + cont + '.bmp'
            newName=UPLOAD_FOLDER + imageType + '/' + imageName
            shutil.move(originalPath, newName)
            result['imageName'] = imageName
            result['imageValue'] = cont
            result['imageType'] = imageType
            logger.info("=====>> uploadImage <<=====   imageName:%s   imageValue:%s", filename, cont)
    return jsonify(result)

@app.route('/judge', methods=['POST'])
def update_file():
    fileName = request.form.get('imageName')
    isRight = request.form.get('right')
    imageType = request.form.get('imageType')
    cont = request.form.get('imageValue')
    logger.info("=====>> judge <<=====   deal error captcha ===>> imageName:%s   imageValue:%s   right:%s", fileName, cont, isRight)
    newName = ''
    #if isRight == '1':
    #    newName=UPLOAD_FOLDER + 'right/' + str(uuid.uuid1()) + '_' + isRight + '_' + cont
    #    shutil.move(UPLOAD_FOLDER + fileName, newName)
    #    result = {
    #        'imageName': fileName,
    #        'newName': newName
    #    }
    #    return jsonify(result)
    #else:
    if isRight == '0':
        newName = str(uuid.uuid1()) + '_0_' +  cont + '.bmp'
        shutil.move(UPLOAD_FOLDER + imageType + '/' + fileName, UPLOAD_ERROR_FOLDER + imageType + '/' + newName)
        result = {
            'imageName': fileName,
            'newName': newName
        }
        return jsonify(result)
    else:
        result = {
            'imageName': fileName,
            'isRight':isRight,
            'newName': '非失败状态(0)的不需要修改'
        }
        return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=False)
