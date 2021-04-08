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
        for a in range(0, 112, 60):
            for b in range(0, 112, 60):
                optical_image = image[i:i + 1, a:a + 400, b:b + 400, :].reshape((400, 400))

                sar_image = sar[i:i + 1, a:a + 400, b:b + 400, :].reshape((400, 400))

                circle_centrio_x = np.random.randint(130, 270)
                circle_centrio_y = np.random.randint(130, 270)

                label_iamge = np.zeros((400, 400), dtype=np.float32)
                rr, cc = draw.circle(circle_centrio_y, circle_centrio_x, 128)
                label_iamge[rr, cc] = 0.2

                rr1, cc1 = draw.circle(circle_centrio_y, circle_centrio_x, 100)
                label_iamge[rr1, cc1] = 0.4

                rr2, cc2 = draw.circle(circle_centrio_y, circle_centrio_x, 60)
                label_iamge[rr2, cc2] = 0.6

                rr3, cc3 = draw.circle(circle_centrio_y, circle_centrio_x, 20)
                label_iamge[rr3, cc3] = 0.8

                rr4, cc4 = draw.circle(circle_centrio_y, circle_centrio_x, 10)
                label_iamge[rr4, cc4] = 1.0

                affine_transform_point1=np.float32([[circle_centrio_y-70,circle_centrio_x],
                                                    [circle_centrio_y+40,circle_centrio_x-50],
                                                    [circle_centrio_y+40,circle_centrio_x+50]])

                affine_transform_point2=np.float32([[circle_centrio_y-70+np.random.randint(-6, 6),circle_centrio_x+np.random.randint(-6, 6)],
                                                    [circle_centrio_y+40+np.random.randint(-6, 6),circle_centrio_x-50+np.random.randint(-6, 6)],
                                                    [circle_centrio_y+40+np.random.randint(-6, 6),circle_centrio_x+50+np.random.randint(-6, 6)]])

                matrix=cv2.getAffineTransform(affine_transform_point1,affine_transform_point2)
                tran_optical_image=cv2.warpAffine(optical_image,matrix,(400,400))
                train_label_image=cv2.warpAffine(label_iamge,matrix,(400,400))

                sub_sar=np.zeros(shape=(400,400),dtype=np.float32)
                sub_sar[rr, cc]=sar_image[rr, cc]


                sub_sub_sar=sub_sar[circle_centrio_y-128:circle_centrio_y+128,circle_centrio_x-128:circle_centrio_x+128]

                sub_sub_sar=np.pad(sub_sub_sar,((72,72),(72,72)),"constant")

                con_image = np.concatenate((tran_optical_image.reshape((400,400,1)), sub_sub_sar.reshape((400,400,1))), axis=2)/255


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