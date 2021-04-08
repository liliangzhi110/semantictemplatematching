import numpy as np
import tensorflow as tf
from skimage import draw
import cv2

filepath = "E:\\competition\\train_dataset3.npz"
image = np.load(filepath)['optical']
sar = np.load(filepath)['sar']
tfrecord_file = 'E:\\fift_paper_dataset\\train_dataset\\val_dataset.tfrecords'

count = 0
with tf.io.TFRecordWriter(tfrecord_file) as writer:
    for i in range(image.shape[0] - 400, image.shape[0]):
        for a in range(0, 112, 60):
            for b in range(0, 112, 60):
                optical_image = image[i:i + 1, a:a + 400, b:b + 400, :].reshape((400, 400))

                sar_image = sar[i:i + 1, a:a + 400, b:b + 400, :].reshape((400, 400))

                circle_centrio_x = np.random.randint(130, 270)
                circle_centrio_y = np.random.randint(130, 270)

                label_iamge = np.zeros((400, 400), dtype=np.float32)
                rr, cc = draw.circle(circle_centrio_y, circle_centrio_x, 128)
                label_iamge[rr, cc] = 1.0

                sub_sar = np.zeros(shape=(400, 400), dtype=np.float32)
                sub_sar[rr, cc] = sar_image[rr, cc]

                sub_sub_sar = sub_sar[circle_centrio_y - 128:circle_centrio_y + 128,
                              circle_centrio_x - 128:circle_centrio_x + 128]

                sub_sub_sar = np.pad(sub_sub_sar, ((72, 72), (72, 72)), "constant")

                con_image = np.concatenate((optical_image.reshape((400, 400, 1)), sub_sub_sar.reshape((400, 400, 1))),
                                           axis=2) / 255

                con_image = con_image.astype(np.float32)
                label = label_iamge.astype(np.float32)



                count = count + 1
                print(count)
                feature = {"train_image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[con_image.tostring()])),
                           "train_label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()]))
                           }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

    writer.close()