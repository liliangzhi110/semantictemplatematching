import numpy as np
import tensorflow as tf
from skimage import draw
import cv2
filepath="E:\\competition\\train_dataset3.npz"
image=np.load(filepath)['optical']
sar=np.load(filepath)['sar']
tfrecord_file = 'E:\\fift_paper_dataset\\train_dataset\\val_dataset.tfrecords'

count=0
with tf.io.TFRecordWriter(tfrecord_file) as writer:
    for i in range(image.shape[0]-400, image.shape[0]):
        for a in range(20, 255, 60):
            for b in range(20, 255, 60):
                optical_image = image[i:i + 1, a:a + 256, b:b + 256, :].reshape((256, 256))
                sar_image = sar[i:i + 1, a:a + 256, b:b + 256, :].reshape((256, 256))

                circle_centrio_x = np.random.randint(90, 166)
                circle_centrio_y = np.random.randint(90, 166)

                label_iamge = np.zeros((256, 256), dtype=np.float32)
                rr, cc = draw.circle(circle_centrio_y, circle_centrio_x, 80)
                label_iamge[rr, cc] = 0.1

                rr1, cc1 = draw.circle(circle_centrio_y, circle_centrio_x, 60)
                label_iamge[rr1, cc1] = 0.4

                rr2, cc2 = draw.circle(circle_centrio_y, circle_centrio_x, 30)
                label_iamge[rr2, cc2] = 0.6

                rr3, cc3 = draw.circle(circle_centrio_y, circle_centrio_x, 15)
                label_iamge[rr3, cc3] = 0.8

                rr4, cc4 = draw.circle(circle_centrio_y, circle_centrio_x, 6)
                label_iamge[rr4, cc4] = 1.0

                affine_transform_point1=np.float32([[circle_centrio_y-60,circle_centrio_x],
                                                    [circle_centrio_y+30,circle_centrio_x-40],
                                                    [circle_centrio_y+30,circle_centrio_x+40]])

                affine_transform_point2=np.float32([[circle_centrio_y-60+np.random.randint(-4, 4),circle_centrio_x+np.random.randint(-4, 4)],
                                                    [circle_centrio_y+30+np.random.randint(-4, 4),circle_centrio_x-40+np.random.randint(-4, 4)],
                                                    [circle_centrio_y+30+np.random.randint(-4, 4),circle_centrio_x+40+np.random.randint(-4, 4)]])

                matrix=cv2.getAffineTransform(affine_transform_point1,affine_transform_point2)
                tran_optical_image=cv2.warpAffine(optical_image,matrix,(256,256))
                train_label_image=cv2.warpAffine(label_iamge,matrix,(256,256))

                sub_sar=np.zeros(shape=(256,256),dtype=np.float32)
                sub_sar[rr, cc]=sar_image[rr, cc]


                sub_sub_sar=sub_sar[circle_centrio_y-80:circle_centrio_y+80,circle_centrio_x-80:circle_centrio_x+80]

                sub_sub_sar=np.pad(sub_sub_sar,((48,48),(48,48)),"constant")

                con_image = np.concatenate((tran_optical_image.reshape((256,256,1)), sub_sub_sar.reshape((256,256,1))), axis=2)/255


                con_image=con_image.astype(np.float32)
                label = train_label_image.astype(np.float32)

                count=count+1
                print(count)
                feature={"train_image":tf.train.Feature(bytes_list=tf.train.BytesList(value=[con_image.tostring()])),
                         "train_label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()]))
                         }
                example=tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

    writer.close()