# -*- coding: utf-8 -*-
import shutil

from panjueshu.panjueshuTrain import crack_captcha_with_file
from external.org_pocoo_werkzeug.werkzeug.utils import secure_filename
from flask import Flask, request, jsonify

app = Flask(__name__)
UPLOAD_FOLDER = 'E://CNTV/'
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
            # shutil.move(UPLOAD_FOLDER + filePath, UPLOAD_FOLDER + 'right/' + con)
    return jsonify(result)


if __name__ == '__main__':
    app.run()
