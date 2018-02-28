# DnCNN-PyTorch

This is a PyTorch implementation of the TIP2017 paper [*Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising*](http://ieeexplore.ieee.org/document/7839189/). The author's [MATLAB implementation is here](https://github.com/cszn/DnCNN).

## How to run

### 1. Dependences
* [PyTorch](http://pytorch.org/)
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
**NOTE**
Experients are still going on. The performance of PyTorch implementation may be further boosted.

### BSD68 Average RSNR

| Noise Level | DnCNN-S | DnCNN-B | DnCNN-S-PyTorch | DnCNN-B-PyTorch |
|:-----------:|:-------:|:-------:|:---------------:|:---------------:|
|     15      |  31.73  |  31.61  |      31.65      |      31.54      |
|     25      |  29.23  |  29.16  |      29.18      |      29.07      |
|     50      |  26.23  |  26.23  |      26.19      |      26.12      |

### Set12 Average PSNR

| Noise Level | DnCNN-S | DnCNN-B | DnCNN-S-PyTorch | DnCNN-B-PyTorch |
|:-----------:|:-------:|:-------:|:---------------:|:---------------:|
|     15      | 32.859  | 32.680  |     32.752      |     32.654      |
|     25      | 30.436  | 30.362  |     30.395      |     30.259      |
|     50      | 27.178  | 27.206  |     27.142      |     27.062      |
