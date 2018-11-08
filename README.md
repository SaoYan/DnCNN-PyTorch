# DnCNN-PyTorch
[![AUR](https://img.shields.io/aur/license/yaourt.svg?style=plastic)](LICENSE)

This is a PyTorch implementation of the TIP2017 paper [*Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising*](http://ieeexplore.ieee.org/document/7839189/). The author's [MATLAB implementation is here](https://github.com/cszn/DnCNN).

****
This code was written with PyTorch<0.4, but most people must be using PyTorch>=0.4 today. Migrating the code is easy. Please refer to [PyTorch 0.4.0 Migration Guide](https://pytorch.org/blog/pytorch-0_4_0-migration-guide/).

****

## How to run

### 1. Dependences
* [PyTorch](http://pytorch.org/)(<0.4)
* [torchvision](https://github.com/pytorch/vision)
* OpenCV for Python
* [HDF5 for Python](http://www.h5py.org/)
* [tensorboardX](https://github.com/lanpa/tensorboard-pytorch) (TensorBoard for PyTorch)

### 2. Train DnCNN-S (DnCNN with known noise level)
```
python train.py \
  --preprocess True \
  --num_of_layers 17 \
  --mode S \
  --noiseL 25 \
  --val_noiseL 25
```
**NOTE**
* If you've already built the training and validation dataset (i.e. train.h5 & val.h5 files), set *preprocess* to be False.
* According to the paper, DnCNN-S has 17 layers.
* *noiseL* is used for training and *val_noiseL* is used for validation. They should be set to the same value for unbiased validation. You can set whatever noise level you need.

### 3. Train DnCNN-B (DnCNN with blind noise level)
```
python train.py \
  --preprocess True \
  --num_of_layers 20 \
  --mode B \
  --val_noiseL 25
```
**NOTE**
* If you've already built the training and validation dataset (i.e. train.h5 & val.h5 files), set *preprocess* to be False.
* According to the paper, DnCNN-B has 20 layers.
* *noiseL* is ingnored when training DnCNN-B. You can set *val_noiseL* to whatever you need.

### 4. Test
```
python test.py \
  --num_of_layers 17 \
  --logdir logs/DnCNN-S-15 \
  --test_data Set12 \
  --test_noiseL 15
```
**NOTE**
* Set *num_of_layers* to be 17 when testing DnCNN-S models. Set *num_of_layers* to be 20 when testing DnCNN-B model.
* *test_data* can be *Set12* or *Set68*.
* *test_noiseL* is used for testing. This should be set according to which model your want to test (i.e. *logdir*).

## Test Results

### BSD68 Average RSNR

| Noise Level | DnCNN-S | DnCNN-B | DnCNN-S-PyTorch | DnCNN-B-PyTorch |
|:-----------:|:-------:|:-------:|:---------------:|:---------------:|
|     15      |  31.73  |  31.61  |      31.71      |      31.60      |
|     25      |  29.23  |  29.16  |      29.21      |      29.15      |
|     50      |  26.23  |  26.23  |      26.22      |      26.20      |

### Set12 Average PSNR

| Noise Level | DnCNN-S | DnCNN-B | DnCNN-S-PyTorch | DnCNN-B-PyTorch |
|:-----------:|:-------:|:-------:|:---------------:|:---------------:|
|     15      | 32.859  | 32.680  |     32.837      |     32.725      |
|     25      | 30.436  | 30.362  |     30.404      |     30.344      |
|     50      | 27.178  | 27.206  |     27.165      |     27.138      |

## Tricks useful for boosting performance
* Parameter initialization:  
Use *kaiming_normal* initialization for *Conv*; Pay attention to the initialization of *BatchNorm*
```
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)
```
* The definition of loss function  
Set *size_average* to be False when defining the loss function. When *size_average=True*, the **pixel-wise average** will be computed, but what we need is **sample-wise average**.
```
criterion = nn.MSELoss(size_average=False)
```
The computation of loss will be like:
```
loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
```
where we divide the sum over one batch of samples by *2N*, with *N* being # samples.
