import numpy as np
import tensorflow as tf
from skimage import draw
import cv2
filepath="E:\\competition\\train_dataset3.npz"
image=np.load(filepath)['optical']
sar=np.load(filepath)['sar']
tfrecord_file = 'E:\\fift_paper_dataset\\train_dataset\\train_dataset.tfrecords'

count=0
with tf.io.TFRecordWriter(tfrecord_file) as writer:
    for i in range(0, image.shape[0]-400):

        optical_image_resize = image[i:i + 1,:,:, :].reshape((512, 512))
        optical_image=cv2.resize(optical_image_resize,(256,256))

        sar_image_resize = sar[i:i + 1,:,:, :].reshape((512, 512))
        sar_image=cv2.resize(sar_image_resize,(256,256))
        for i in range(20):

            circle_centrio_x = np.random.randint(0, 90)
            circle_centrio_y = np.random.randint(0, 90)

            label_iamge = np.zeros((256, 256), dtype=np.float32)

            label_iamge[circle_centrio_y:circle_centrio_y+160, circle_centrio_x:circle_centrio_x+160] = 1.0

            # rr1, cc1 = draw.circle(circle_centrio_y, circle_centrio_x, 60)
            # label_iamge[rr1, cc1] = 0.4
            #
            # rr2, cc2 = draw.circle(circle_centrio_y, circle_centrio_x, 30)
            # label_iamge[rr2, cc2] = 0.6
            #
            # rr3, cc3 = draw.circle(circle_centrio_y, circle_centrio_x, 15)
            # label_iamge[rr3, cc3] = 0.8
            #
            # rr4, cc4 = draw.circle(circle_centrio_y, circle_centrio_x, 6)
            # label_iamge[rr4, cc4] = 1.0

            sub_sar=sar_image[circle_centrio_y:circle_centrio_y+160, circle_centrio_x:circle_centrio_x+160]

            sub_sub_sar = np.pad(sub_sar, ((48, 48), (48, 48)), "constant")

            con_image = np.concatenate((optical_image.reshape((256,256,1)), sub_sub_sar.reshape((256,256,1))), axis=2)/255


            con_image=con_image.astype(np.float32)
            label = label_iamge.astype(np.float32)

            cv2.imwrite("E:\\fift_paper_dataset\\train_dataset\\display\\"+str(count)+"_optical.tif",optical_image)
            cv2.imwrite("E:\\fift_paper_dataset\\train_dataset\\display\\"+str(count)+"_sar.tif",sub_sub_sar)
            cv2.imwrite("E:\\fift_paper_dataset\\train_dataset\\display\\"+str(count)+"_label.tif",label_iamge)

            count=count+1
            print(count)
            feature={"train_image":tf.train.Feature(bytes_list=tf.train.BytesList(value=[con_image.tostring()])),
                     "train_label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()]))
                     }
            example=tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    writer.close()