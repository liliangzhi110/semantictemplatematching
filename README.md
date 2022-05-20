# A semantic template matching framework for remote sensing image registration  



![image-20210408200648767](https://github.com/liliangzhi110/semantictemplatematching/blob/main/img/frame.jpg)



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

## Example of a training data

![Framework](https://github.com/liliangzhi110/semantictemplatematching/blob/main/img/102_optical.png)       ![image-20210408211154158](https://github.com/liliangzhi110/semantictemplatematching/blob/main/img/102_sar.png)           ![image-20210408211354390](https://github.com/liliangzhi110/semantictemplatematching/blob/main/img/102_label.png)         
reference image----------------------,---------------------template image,---------------------------,label

## Get started

First download the training data, place it under the project, and then generate the .tfrecord file using the code in generate data.

## Training scripts

Training semantic template matching models using model files
## ISPRS paper
Li L, Han L, Ding M, et al. A deep learning semantic template matching framework for remote sensing image registration[J]. ISPRS Journal of Photogrammetry and Remote Sensing, 2021, 181: 205-217.

