# semantic template matching framework for remote sensing image registration  



![image-20210408200648767](C:\Users\lilia\AppData\Roaming\Typora\typora-user-images\image-20210408200648767.png)



TensorFlow implementation of semantic template matching.



## Requirements

Please use Python 3.7, install NumPy, OpenCV (3.4.2) and TensorFlow (2.0.0). 

## train sample

| data               |                                                              |
| ------------------ | ------------------------------------------------------------ |
| google earth image | https://drive.google.com/drive/folders/1LV8n80daRKCySmCCQ1nZP6lB4aRsN3CM?usp=sharing |
| landsat-8          |                                                              |
| GF-2               |                                                              |
| SAR-optical        | https://drive.google.com/drive/folders/12x2m2temb5IdUjUfhWuEzCK1sXXT2ZME?usp=sharing |

![image-20210408211137145](\0_optical.tif)       ![image-20210408211154158](C:\Users\lilia\AppData\Roaming\Typora\typora-user-images\image-20210408211154158.png)           ![image-20210408211354390](C:\Users\lilia\AppData\Roaming\Typora\typora-user-images\image-20210408211354390.png)                      

reference image                                                    template image                                         label

## Get started

First download the training data, place it under the project, and then generate the .tfrecord file using the code in generate data.

## Training scripts

Training semantic template matching models using model files

