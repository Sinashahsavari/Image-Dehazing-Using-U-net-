# Image Dehazing Using U-net Architecture with Wavelet Feature Extraction

## Description 

This is project "Image Dehazing Using U-net Architecture with Wavelet Feature Extraction" developed by Sina Shahsavari, Ayman jabaren.

## Abstract 

Image dehazing is an ill-posed problem challenging many computer vision applications. It is an undesirable physical phenomena that occurs while capturing real pictures. Many methods and algorithms were developed for that end. Nonetheless, no generic high-quality solution has been proposed yet. Hence, we are proposing an end-to-end architecture that tackles the problem and outperforms state-of-the-art approaches. Deep network has shown promising results when it comes to image dehazing. We investigated and harnessed wavelet filterbank properties for feature extraction in U-net neural network. By doing so, we have combined U-net learning properties with image specific efficient features extraction. While enhancing the model performance, we explored the effect of different types of wavelet filterbanks as well as different number of wavelet levels on the quality of image effecting.

Index Terms— Wavelet filterbank, feature extraction, multi-level wavelet, U-net, image dehazing,

## For more details please read our final paper 
- https://github.com/Sinashahsavari/Image-Dehazing-Using-U-net-/blob/master/final%20paper.pdf

## Requirements and Usage
### Dependencies
* [Python 3.6+](https://www.continuum.io/downloads)
* [PyTorch 0.4.0+](http://pytorch.org/)
* [PyWt](https://pypi.org/project/PyWt/)

<br/>

### Downloading the [RESIDE](http://t.cn/RQXyZFI ) dataset
We only use images in clear and haze folders
All claar images are divided as training images (train_clear), testing images (train_hazy).
The hazy images should be placed to corresponding folders (val_clear and val_hazy).

###  Training
```bash
$ CUDA_VISIBLE_DEVICES=0 python train.py --epochs 100 \
                --lr 1e-4 \
                --use_gpu true \
                --gpu 0 \
                --ori_data_path /train_clear/ \
                --haze_data_path /train_hazy \
                --val_ori_data_path /val_clear/ \
                --val_haze_data_path /val_hazy/ \
                --num_workers 4 \
                --batch_size 10 \
                --val_batch_size 2 \
                --print_gap 500 \
                --model_dir /model/ \
                --log_dir /model/ \
                --sample_output_folder /samples/ \
                --net_name /dehaze_chromatic_
```
### Experiment1
In wavelet based image dehazing the selection of wavelets is an essential task which determines the dehazed image quality. There are several choices of wavelets with different properties. Therefore, in the first experiment, we aim to investigate the different wavelet types performance in image dehazing and features extraction when employed in the model described in the paper. We use different bases of wavelet filters. We have used Haar, Daubechies, Symlets and Biorthogonal wavelets. we are interested in finding whether certain wavelets outperform others and investigate their properties and advantages in image dehazing U-net. Some of the properties that we think might affect the performance are: Orthogonality, symmetry, number of vanishing moments, power perseverance and filter size.



### Experiment2
In this experiment, we aim to enhance image dehazing by boosting feature extraction using multiple levels wavelets filterbanks. Therefore, we have replaced the first layer’s Haar wavelet filterbank by a two level Haar wavelet filterbank. By applying a two level wavelet feature bank, we would split the input image to narrower frequency sub-bands in the low frequencies where most of the crucial image and haze features are located.


## Code organization 

### experiment1 and experiment2

 - experiment1: exploring the effect of different types of wavelets in the U-net
 - experiment2: exploring the effect of different nuumber of wavelets levels in the U-net
 - experiment1/config.yml: Configuration file
 - experiment1/PSNR.yml: Configuration file
 - experiment1/wavelet.yml: contains the wavelet feature extraction functions as well as inverse wavelet functions for reconstruction  
 - experiment1/train.py: Module for training and evaluation
 - experiment1/model.py: Module for Model, which implements U-net network with wavelet feature extraction layers
 - experiment1/data: Data files
 - experiment1/Utils: some used functions  
 - experiment1/README.md: Information for experiment1





## Refrences
This work is mainly based on this paper:
[Wavelet U-net for image dehazing](https://ieeexplore.ieee.org/document/8803391)



