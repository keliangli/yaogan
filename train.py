# coding: utf-8
# 读取.tfrecords格式数据集，进行geture的cnn构建、训练、模型保存
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import 遥感.vgg_16_net as vgg_16_net
import 遥感.deep_net_1 as  deep_net_1


train_whole_sample_size = 13000  # 训练集总量
test_whole_sample_size = 2000  # 测试集总量
gesture_class = 6
test_batch_size = 10  # 测试集每个批次的样本个数
train_batch_size = 20

image_channel = 3
image_size = 600

# 训练集.tfrecords 路径
train_path = r"G:\likeliang_data\test_1\test_1\遥感\train_WHU.tfrecords"
test_path = r"G:\likeliang_data\test_1\test_1\遥感\test_WHU.tfrecords"
# tensorboard的graph文件 保存路径
graph_path = r"G:\likeliang_data\test_1\test_1\dog_and_cat\graph"
# CNN模型文件 保存路径
cnn_model_save_path = r"G:\likeliang_data\test_1\test_1\dog_and_cat\cnn_model\cnn_model.ckpt"

print("/****************************/")
print("   gesture cnn train ~~~")


# function: 解码 .tfrecords文件
def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string),
        }
    )
    img = tf.decode_raw(features['img_raw'], tf.uint8)  # 原图数据二进制解码为 无符号8位整型
    img = tf.reshape(img, [image_size, image_size, image_channel])
    img = tf.cast(img, tf.float32) * (1. / 255.) - 0.5  # 数据归一化
    label = tf.cast(features['label'], tf.int64)  # 获取样本对应的标签
    return img, label  # 返回样本及对应的标签


print("step 1 ~~~")

# function:加载tfrecords文件并进行文件解析
# train batch 训练集
img_train, labels_train = read_and_decode(train_path)

# 定义模型训练时的数据批次
img_train_batch, labels_train_batch = tf.train.shuffle_batch([img_train, labels_train],
                                                             batch_size=train_batch_size,
                                                             capacity=train_whole_sample_size,
                                                             min_after_dequeue=1000,
                                                             )

train_labes = tf.one_hot(labels_train_batch, gesture_class, 1, 0)  # label转为 one_hot格式


# test batch 测试集
img_test, labels_test = read_and_decode(test_path)

img_test_batch, labels_test_batch = tf.train.shuffle_batch([img_test, labels_test],
                                                           batch_size=test_batch_size,
                                                           capacity=test_whole_sample_size,
                                                           min_after_dequeue=1000,
                                                           )

test_labes = tf.one_hot(labels_test_batch, gesture_class, 1, 0)  # label转为 one_hot格式


print("step 2 ~~~")

x = tf.placeholder(tf.float32, [None, image_size, image_size, image_channel], name="images")  # 注意：x的shape
y = tf.placeholder(tf.float32, [None, gesture_class], name="labels")
keep_prob = tf.placeholder(tf.float32, name="my_keep_prob")


# shape : 4D
def weight_variable(shape, f_name):
    initial = tf.truncated_normal(shape, mean=0, stddev=0.1)  # 生成截断的正太分布
    return tf.Variable(initial, name=f_name)


# 初始化偏置
def bias_variable(shape, f_name):
    initial = tf.constant(0.1, shape=shape)  # 生成截断的正太分布
    return tf.Variable(initial, name=f_name)


# 卷积层
def Conv2d_Filter(x, W,strides):

    return tf.nn.conv2d(x, W, strides=strides, padding="SAME")


# max-pooling 池化层
def max_pooling_2x2(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 卷积层 1
with tf.name_scope('Conv1'):
    W_conv1 = weight_variable([20, 20, 3, 32], 'W_conv1')  # 3通道输入32通道输出

    b_conv1 = bias_variable([32], 'b_conv1')  # 32个输出对应32个偏置

    with tf.name_scope('h_conv1'):

        h_conv1 = tf.nn.relu(Conv2d_Filter(x, W_conv1,[1,20,20,1]) + b_conv1)

# 池化层 1
with tf.name_scope('Pool1'):
    h_pool1 = max_pooling_2x2(h_conv1)

# 卷积层 2
with tf.name_scope('Conv2'):
    W_conv2 = weight_variable([5, 5, 32, 16], 'W_conv2')

    b_conv2 = bias_variable([16], 'b_conv2')

    with tf.name_scope('h_conv2'):  # 把h_pool1通过卷积操作，加上偏置值，应用于 relu函数激活
        h_conv2 = tf.nn.relu(Conv2d_Filter(h_pool1, W_conv2,[1,1,1,1]) + b_conv2)

# 全连接层 1
with tf.name_scope('Fc1'):
    # 初始化第一个全连接层权值
    W_fc1 = weight_variable([15 * 15 * 16, 512], 'W_fc1')

    b_fc1 = bias_variable([512], 'b_fc1')  # 1024个节点

    # 池化层的输出扁平化，变为1维张量
    with tf.name_scope('Pool2_flat'):
        h_pool4_flat = tf.reshape(h_conv2, [-1, 15 * 15 * 16])
    # 全连接层的输出
    with tf.name_scope('h_fc1'):
        h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

    # keep_prob 用来表示神经元输出的更新概率
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name="my_h_fc1_drop")

# 全连接层 2
with tf.name_scope('Fc2'):
    # 第二个全连接层
    W_fc2 = weight_variable([512, gesture_class], 'W_fc2')

    b_fc2 = bias_variable([gesture_class], 'b_fc2')

    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name="my_prediction")


# 交叉熵代价函数
with tf.name_scope('Corss_Entropy'):
    corss_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction),
                                   name="loss")  # 交叉熵

# 使用Adam优化器进行迭代
with tf.name_scope('Train_step'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(corss_entropy, name="train_step")

# 统计真实分类 和 预测分类
correct_prediction = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# function: 训练模型
print("cnn train start ~~~")
#
with tf.Session() as sess:  # 开始一个会话

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()  # 协同启动的线程
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)  # 启动线程运行队列

    saver = tf.train.Saver()  # 模型保存
    max_acc = 0  # 最高测试准确率测试

    for i in range(20001):

        img_xs, label_xs = sess.run([img_train_batch, train_labes])  # 读取训练 batch

        sess.run(train_step, feed_dict={x: img_xs, y: label_xs, keep_prob: 0.75})

        if (i % 1000) == 0:
            print("训练第", i, "次")
            img_test_xs, label_test_xs = sess.run([img_test_batch, test_labes])  # 读取测试 batch
            acc = sess.run(accuracy, feed_dict={x: img_test_xs, y: label_test_xs, keep_prob: 1.0})
            print("Itsers = " + str(i) + "  准确率: " + str(acc))


            if max_acc < acc:  # 记录测试准确率最大时的模型
                max_acc = acc
                saver.save(sess, save_path=cnn_model_save_path)



    coord.request_stop()
    coord.join(threads)
    sess.close()

