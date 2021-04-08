import numpy as np
import tensorflow as tf
from skimage import draw
import cv2
filepath="E:\\fift_paper_dataset\\googleimage_xian\\image.npz"
im=np.load(filepath)['image']

image=im[:,:,0:1]
sar=im[:,:,1:2]
tfrecord_file = 'E:\\fift_paper_dataset\\train_dataset\\val_dataset.tfrecords'

count=0
with tf.io.TFRecordWriter(tfrecord_file) as writer:

    for a in range(10000, 11000, 50):
        for b in range(10000, 11000, 50):
            optical_image = image[a:a + 256, b:b + 256, :].reshape((256, 256))
            sar_image = sar[a:a + 256, b:b + 256, :].reshape((256, 256))

            circle_centrio_x = np.random.randint(90, 166)
            circle_centrio_y = np.random.randint(90, 166)

            label_iamge=np.zeros((256,256),dtype=np.float32)

            label_iamge[circle_centrio_y-64:circle_centrio_y+64,circle_centrio_x-64:circle_centrio_x+64]=1.0

            affine_transform_point1=np.float32([[circle_centrio_y,circle_centrio_x],
                                                [circle_centrio_y+30,circle_centrio_x-40],
                                                [circle_centrio_y+30,circle_centrio_x+40]])

            affine_transform_point2=np.float32([[circle_centrio_y,circle_centrio_x],
                                                [circle_centrio_y+30+np.random.randint(-4, 4),circle_centrio_x-40+np.random.randint(-4, 4)],
                                                [circle_centrio_y+30+np.random.randint(-4, 4),circle_centrio_x+40+np.random.randint(-4, 4)]])

            matrix=cv2.getAffineTransform(affine_transform_point1,affine_transform_point2)
            tran_optical_image=cv2.warpAffine(optical_image,matrix,(256,256))


            sub_sar=np.zeros(shape=(256,256),dtype=np.float32)
            sub_sar[64:64+128,64:128+64]=sar_image[circle_centrio_y-64:circle_centrio_y+64,circle_centrio_x-64:circle_centrio_x+64]


            con_image = np.concatenate((tran_optical_image.reshape((256,256,1)), sub_sar.reshape((256,256,1))), axis=2)/255


            con_image=con_image.astype(np.float32)
            label = label_iamge.astype(np.float32)



            count=count+1
            print(count)
            feature={"train_image":tf.train.Feature(bytes_list=tf.train.BytesList(value=[con_image.tostring()])),
                     "train_label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()]))
                     }
            example=tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    writer.close()