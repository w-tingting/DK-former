# DK-former:Deep Kernel Gaussian Process Transformer Network for Traffic Sign Recognition in Autonomous Vehicles

## Introdction

 Implementation of DK-GPT, a new deep network DK-GPT for traffic sign recognition is proposed by using convolutional random Fourier features, which makes full use of the advantages of the kernel method for nonlinear small sample classification and transformer structure, and realizes the effective extraction of local and global information.

 **Note: This code only provides the core implementation of the network architecture, and the complete code will be made public after the paper is published.**

![](https://github.com/w-tingting/image-folder/blob/main/%E6%80%BB%E4%BD%93%E7%BD%91%E7%BB%9C%E6%9E%B6%E6%9E%841.jpg)

​                                Fig.1 The overall framework of deep kernel Gaussian process transformer network.

##  Installation

 

1. Ubuntu 18.04
2. python 3.8.18
3. pytorch 2.1.0
4. pytorch_warmup
5. apex 

## Results

 A.  Comparison

  Table 1 Comparison results (in %) of different traffic sign recognition algorithms

|        method        | GTSRB     | Indian    | Chinese   |
| :------------------: | --------- | --------- | --------- |
|         KNN          | 86.77     | 79.15     | 82.94     |
|         SVM          | 96.36     | 92.71     | 94.29     |
|    Random Forest     | 97.20     | 94.12     | 91.78     |
|         ViT          | 98.40     | 97.45     | 97.41     |
|         CaiT         | 96.96     | 89.25     | 90.44     |
|       CrossViT       | 98.63     | 83.92     | 92.28     |
| Sinkhorn transformer | 97.04     | 94.02     | 85.61     |
|    Nystromformer     | 83.15     | 80.13     | 79.08     |
|         TNT          | 97.73     | 92.75     | 94.52     |
|       DeepViT        | 97.29     | 83.92     | 93.08     |
|       AlexNet        | 96.61     | 98.14     | 97.78     |
|      MobileNet       | 98.57     | 93.41     | 94.08     |
|      shuffleNet      | 98.36     | 98.48     | 99.08     |
|        RFFNet        | 78.80     | 69.76     | 58.11     |
|      **DK-GPT**      | **99.09** | **99.32** | **99.23** |

B. Grad_Cam

<img width="150" height="150" src="https://github.com/w-tingting/image-folder/blob/main/GTSRB_30_gradcam.png" alt="GTSRB_img_slow_gradcam" style="zoom:10%;" />

<img width="150" height="150" src="https://github.com/w-tingting/image-folder/blob/main/GTSRB_token_30_gradcam.png" alt="data_china_token_slow_gradcam" style="zoom:10%;" />

<img width="150" height="150" src="https://github.com/w-tingting/image-folder/blob/main/GTSRB_block_30_gradcam.png" alt="data_china_block_slow_gradcam" style="zoom:10%;" />

<img width="150" height="150" src="https://github.com/w-tingting/image-folder/blob/main/GTSRB_head_30_gradcam.png" alt="GTSRB_head_slow_gradcam" style="zoom:10%;" />
