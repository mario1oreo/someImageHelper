import datetime
import numpy as  np
import matplotlib.pyplot as  plt
from PIL import Image, ImageDraw
import random
import tensorflow as tf
import os

number = ['2', '3', '4', '5', '6', '7', '8']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'm', 'n', 'p',  'w', 'x', 'y']
Alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

char_set = number + alphabet

##图片高
IMAGE_HEIGHT = 50
##图片宽
IMAGE_WIDTH = 200
##验证码长度
MAX_CAPTCHA = 5
##验证码选择空间
CHAR_SET_LEN = len(char_set)
##提前定义变量空间
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  ##节点保留率


# 根据一个点A的灰度值(0/255值),与周围的8个点的值比较
# 降噪率N: N=1,2,3,4,5,6,7
# 当A的值与周围8个点的相等数小于N时,此点为噪点
# 如果确认是噪声,用该点的上面一个点的值进行替换
def get_near_pixel(image, x, y, N):
    pix = image.getpixel((x, y))

    near_dots = 0
    if pix == image.getpixel((x - 1, y - 1)):
        near_dots += 1
    if pix == image.getpixel((x - 1, y)):
        near_dots += 1
    if pix == image.getpixel((x - 1, y + 1)):
        near_dots += 1
    if pix == image.getpixel((x, y - 1)):
        near_dots += 1
    if pix == image.getpixel((x, y + 1)):
        near_dots += 1
    if pix == image.getpixel((x + 1, y - 1)):
        near_dots += 1
    if pix == image.getpixel((x + 1, y)):
        near_dots += 1
    if pix == image.getpixel((x + 1, y + 1)):
        near_dots += 1

    if near_dots < N:
        # 确定是噪声,用上面一个点的值代替
        return image.getpixel((x, y - 1))
    else:
        return None


# 降噪处理
def clear_noise(image, N):
    draw = ImageDraw.Draw(image)

    # 外面一圈变白色
    Width, Height = image.size
    for x in range(Width):
        draw.point((x, 0), 255)
        draw.point((x, Height - 1), 255)
    for y in range(Height):
        draw.point((0, y), 255)
        draw.point((Width - 1, y), 255)

    # 内部降噪
    for x in range(1, Width - 1):
        for y in range(1, Height - 1):
            color = get_near_pixel(image, x, y, N)
            if color != None:
                draw.point((x, y), color)


def initTable():
    table = []
    for i in range(256):
        if i < 75:
            table.append(0)
        else:
            table.append(1)
    return table


def preHandleImage(imagePath):
    img = Image.open(imagePath)
    img = img.convert('L')
    img = img.point(initTable(), '1')
    img = img.convert('L')
    clear_noise(img, 4)
    img = np.array(img)
    img = 1 * (img.flatten())
    return img


# fileDir = 'E:/work/captcha/genSusong/bj/traindeal/'
# fileDir = 'E:/work/captcha/genSusong/bj/testdeal/'
# fileDir = 'E:/work/captcha/genSusong/bj/temp/'
# all_image = os.listdir(fileDir)
imageContain = []
# for i in range(0, len(all_image) - 1):
#     base = os.path.basename(fileDir + all_image[i])
#     name = os.path.splitext(base)[0]
#     image = Image.open(fileDir + all_image[i])
#     # image = image.convert('L')
#     # image = image.point(initTable(), '1')
#     # image = image.convert('L')
#     # clear_noise(image, 4)
#     image = np.array(image)
#     image = 1 * (image.flatten())
#     imageContain.append([name, image])
#     if i % 1000 == 0:
#         print(datetime.datetime.now())
#         print('loading  ======>>> ' + str(i))
# print('loading  finished!')


##使用ImageCaptcha库生成验证码
def gen_captcha_text_and_image(fileDir='E:/work/captcha/genSusong/train/'):
    maxRandom = len(imageContain) - 1
    random_file = random.randint(0, maxRandom)
    captcha_name = imageContain[random_file][0]
    captcha_image = imageContain[random_file][1]
    return captcha_name, captcha_image


##彩色图转化为灰度图
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # print(gray)
        return gray
    else:
        return img


##获取字符在 字符域中下标
def getPos(char_set=char_set, char=None):
    return char_set.index(char)


##验证码字符转换为长向量
def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长' + str(MAX_CAPTCHA) + '个字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + getPos(char=c)
        vector[idx] = 1
    return vector


def vec2text(vec, char_set=char_set):
    text = ''
    if len(vec) > MAX_CAPTCHA:
        raise ValueError('验证码最长' + str(MAX_CAPTCHA) + '个字符')
    for i in vec:
        text = text + char_set[i]
    return text


##获得1组验证码数据
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    for i in range(batch_size):
        text, image = gen_captcha_text_and_image()
        image = convert2gray(image)
        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text2vec(text)
        # print(text)
        # print(text2vec(text))
    return batch_x, batch_y


##卷积层 附relu  max_pool drop操作
def conn_layer(w_alpha=0.01, b_alpha=0.1, _keep_prob=0.75, input=None, last_size=None, cur_size=None):
    w_c1 = tf.Variable(w_alpha * tf.random_normal([7, 7, last_size, cur_size]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([cur_size]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob=_keep_prob)
    return conv1


##对卷积层到全链接层的数据进行变换
def _get_conn_last_size(input):
    shape = input.get_shape().as_list()
    dim = 1
    for d in shape[1:]:
        dim *= d
    input = tf.reshape(input, [-1, dim])
    return input, dim


##全链接层
def _fc_layer(w_alpha=0.01, b_alpha=0.1, input=None, last_size=None, cur_size=None):
    w_d = tf.Variable(w_alpha * tf.random_normal([last_size, cur_size]))
    b_d = tf.Variable(b_alpha * tf.random_normal([cur_size]))
    fc = tf.nn.bias_add(tf.matmul(input, w_d), b_d)
    return fc


##构建前向传播网络
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    conv1 = conn_layer(input=x, last_size=1, cur_size=32)
    conv2 = conn_layer(input=conv1, last_size=32, cur_size=64)
    conn3 = conn_layer(input=conv2, last_size=64, cur_size=128)

    input, dim = _get_conn_last_size(conn3)

    fc_layer1 = _fc_layer(input=input, last_size=dim, cur_size=1024)
    fc_layer1 = tf.nn.relu(fc_layer1)
    fc_layer1 = tf.nn.dropout(fc_layer1, keep_prob)

    fc_out = _fc_layer(input=fc_layer1, last_size=1024, cur_size=MAX_CAPTCHA * CHAR_SET_LEN)
    return fc_out


##反向传播
def back_propagation():
    output = crack_captcha_cnn()
    ##学习率
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
    optm = tf.train.AdamOptimizer(1e-4).minimize(loss)
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.arg_max(predict, 2)
    max_idx_l = tf.arg_max(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(max_idx_p, max_idx_l), tf.float32))
    return loss, optm, accuracy


##初次运行训练模型
def train_first():
    loss, optm, accuracy = back_propagation()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        baseAcc = 0.5
        while 1:
            batch_x, batch_y = get_next_batch(256)
            _, loss_ = sess.run([optm, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.25})
            # if step % 25 == 0:
            #     print(step, loss_)
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(400)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(step, acc, loss_, datetime.datetime.now())
                if acc > baseAcc:  ##准确率大于0.80保存模型 可自行调整
                    baseAcc = acc
                    saver.save(sess, './susongwuyouBj.model', global_step=step)
                elif acc > 0.95:  # or (baseAcc > 0.9 and baseAcc > acc):
                    saver.save(sess, './susongwuyouBj.model', global_step=step)
                    break
            step += 1


##加载现有模型 继续进行训练
def train_continue(step):
    loss, optm, accuracy = back_propagation()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        path = "./susongwuyouBj.model-" + str(step)
        saver.restore(sess, path)
        ##36300 36300 0.9325 0.0147698
        while 1:
            batch_x, batch_y = get_next_batch(128)
            _, loss_ = sess.run([optm, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.65})
            # if step % 20 == 0:
            #     print(step, loss_)
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(200)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(step, acc, loss_, datetime.datetime.now())
                if acc >= 0.925:
                    saver.save(sess, './susongwuyouBj.model', global_step=step)
                if acc >= 0.95:
                    saver.save(sess, './susongwuyouBj.model', global_step=step)

                    break
            step += 1


output = crack_captcha_cnn()
saver = tf.train.Saver()
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
# with tf.Session() as sess:
# path = './susongwuyou.model-1200'
saver.restore(sess, 'E:\\work\\pythonWorkspace\\someImageHelper\\susongwuyouBJ\\susongwuyouBj.model-3100')


##测试训练模型
def crack_captcha(captcha_image):
    imag = preHandleImage(captcha_image)
    predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    text_list = sess.run(predict, feed_dict={X: [imag], keep_prob: 1})
    text = text_list[0].tolist()
    result = vec2text(text)
    # right = False
    # if result in captcha_image:
    #     right = True
    # print(captcha_image, result, right)
    return result


# rightCounter = 0
# for i in range(0, 1000):
#     result, right = crack_captcha(fileDir + all_image[random.randint(0, len(all_image) - 1)])
#     if right:
#         rightCounter = rightCounter + 1
# print('1000次测试成功率：', str(rightCounter / 1000))

# train_first()

