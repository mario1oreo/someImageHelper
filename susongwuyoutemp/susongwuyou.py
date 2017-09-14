import numpy as  np
import tensorflow as tf
from PIL import Image

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
Alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

char_set = number

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


def initTable():
    table = []
    for i in range(256):
        if i < 90:
            table.append(0)
        else:
            table.append(1)
    return table


def preHandleImage(imagePath):
    im = Image.open(imagePath)
    im = im.convert('L')
    img = im.point(initTable(), '1')
    img = np.array(img)
    img = 1 * (img.flatten())
    return img


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
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    """
    def char2pos(c):  
        if c =='_':  
            k = 62  
            return k  
        k = ord(c)-48  
        if k > 9:  
            k = ord(c) - 55  
            if k > 35:  
                k = ord(c) - 61  
                if k > 61:  
                    raise ValueError('No Map')   
        return k  
    """
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + getPos(char=c)
        vector[idx] = 1
    return vector


##卷积层 附relu  max_pool drop操作
def conn_layer(w_alpha=0.01, b_alpha=0.1, _keep_prob=0.75, input=None, last_size=None, cur_size=None):
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, last_size, cur_size]))
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
    optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.arg_max(predict, 2)
    max_idx_l = tf.arg_max(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(max_idx_p, max_idx_l), tf.float32))
    return loss, optm, accuracy


output = crack_captcha_cnn()
saver = tf.train.Saver()
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
# with tf.Session() as sess:
path = './susongwuyou.model-1200'
saver.restore(sess, path)


##测试训练模型
def crack_captcha(captcha_image):
    imag = preHandleImage(captcha_image)
    predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    text_list = sess.run(predict, feed_dict={X: [imag], keep_prob: 1})
    text = text_list[0].tolist()
    result = ''.join(map(str, text))
    return result
