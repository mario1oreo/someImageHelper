#!flask/bin/python
from panjueshu.panjueshuTrain import *
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

logger = logging.getLogger('判决书验证码')  # 获取名为tst的logger
logger.addHandler(handler)  # 为logger添加handler
logger.setLevel(logging.INFO)




app = Flask(__name__)
UPLOAD_FOLDER = '/usr/local/spider/image/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'bmp'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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
            file.save(app.config['UPLOAD_FOLDER'] + filename)
            cont = crack_captcha_with_file(app.config['UPLOAD_FOLDER'] + filename)
            result['imageName'] = filename
            result['imageValue'] = cont
            logger.info("=====>> uploadImage <<=====   imageName:%s   imageValue:%s", filename, cont)
    return jsonify(result)

@app.route('/judge', methods=['POST'])
def update_file():
    filePath = request.form.get('imageName')
    isRight = request.form.get('right')
    con = request.form.get('imageValue')+'.bmp'
    logger.info("=====>> judge <<=====   imageName:%s   imageValue:%s   right:%s", filePath, con, isRight)
    newName = ''
    if isRight == '1':
        newName=UPLOAD_FOLDER + 'right/' + str(uuid.uuid1()) + '_' + isRight + '_' + con
        shutil.move(UPLOAD_FOLDER + filePath, newName)
        result = {
            'imageName': filePath,
            'newName': newName
        }
        return jsonify(result)
    else:
        newName=UPLOAD_FOLDER + 'error/' + str(uuid.uuid1()) + '_' + isRight + '_' +  con
        shutil.move(UPLOAD_FOLDER + filePath, newName)
        result = {
            'imageName': filePath,
            'newName': newName
        }
        return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=False)
