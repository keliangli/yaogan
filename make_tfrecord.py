import os
import tensorflow as tf
from PIL import Image

cwd = os.getcwd()



def create_record(imgfile_path,record_name):

    imgfile_path = imgfile_path
    os.chdir(imgfile_path)
    writer = tf.python_io.TFRecordWriter(record_name+".tfrecords")

    index = 0

    for name in os.listdir(imgfile_path):
        print(name[-3:])
        if name[-3:] == 'rds':
            continue
        class_path = imgfile_path +"\\"+name+"\\"
        os.chdir(class_path)
        print(class_path)

        for img_name in os.listdir(class_path):
            if img_name[-1] != 'g':
                continue
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((600, 600))
            img_raw = img.tobytes() #将图片转化为原生bytes
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(name)])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
        index = index + 1
        os.chdir(imgfile_path)

    writer.close()


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [600, 600, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label




if __name__ == '__main__':

    print("make test data")
    imgfile_path = r"G:\likeliang_data\cifar-10-python\遥感图片\test_data"
    name = "test_WHU"
    create_record(imgfile_path,name)

    print("make train data")
    imgfile_path = r"G:\likeliang_data\cifar-10-python\遥感图片\train_data"
    name = "train_WHU"
    create_record(imgfile_path,name)
