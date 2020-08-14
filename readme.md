## Semantic Randomized Smoothing
---

*Browse our paper on [arXiv](https://arxiv.org/abs/2002.12398)!*

Fork from Cohen et al's randomized smoothing code framework [(link)](https://github.com/locuslab/smoothing).

We support semantic transformations including Gaussian blur, translation, brightness and contrast, rotation and scaling.

The implementation is based on PyTorch framework, and requires GPU support. We recommend to run the code on GPU no lower than NVIDIA GTX 1080 Ti level. Before running the code, please install all dependencies according to `requirements.txt`.

### File Structure

The root folder is created from Cohen et al's `code/` folder. Our new code (for semantic transformation certified robustness triaining and certifying) is mainly placed in `semantic/` folder.

In the following we only list changed or new files:

- `requirements.txt`: list of required packages
- `architectures.py`: add model support on MNIST dataset
- `certify.py`: improve data storage process and real-time progress showing for $\ell_2$ certification
- `datasets.py`: add MNIST dataset support
- `train.py`: improve data storage process for $\ell_2$ certification
- `archs/`
  - `mnist_conv.py`: **(new)** define the model structure of MNIST convnet.
- `data/`: **(new)** experiment data; format explained in detail later
- `figures/`: **(new)** 
  - `bound_compare*.pdf`: different noise distributions theoretical bound comparison
  - other `*.pdf`: curves of robust accuracy for comparison
  - `cifar10/`: example CIFAR-10 images of each semantic transformation
  - `mnist/`: example MNIST images of each semantic transformation 
- `semantic/`: **(new)** 
  - `__init__.py`: blank
  - `transforms.py`: Implementation of semantic transformations. For some time-costly transformations like rotation and scaling, we accelerate them by tensor computation.
  - `transformer.py`: Compose the parameter sampling, image transformation, and robustness guarantee computation. Each combination is implemented by a `AbstractTransfromer` class and used in `certify.py` and `train.py`.
  - `core.py`: Randomized smoothing computation core classes.
  - `train.py`: Train models using corresponding data augmentation.
  - `certify.py`: Randomized smoothing based certification.
  - `strict_rotation_aliasing_analyze`: Compute the maximum $\ell_2$ aliasing for rotation samples for each input image using our Lipschitz bounding based technique.
  - `strict_resize_aliasing_analyze`: Compute the maximum $\ell_2$ aliasing for scaling(i.e., resize) samples for each input image using our Lipschitz bounding based technique.
  - `strict_rotation_certify.py`: Compute the sampling-based certification for rotation.
  - `strict_resize_certify.py`: Compute the sampling-based certification for scaling(i.e., resize).
  - `translation_certify.py`: Enumeration-based translation certification.
  - `analyze.py`: Analyze the output of above cerfication. Then output general reader-friendly stats.
  - `helper/`:
    - `parallel_runner.py`: Run sampling-based aliasing computation and certification in parallel by partitioning datasets and creating processes for each subset.
    - `result_concate.py`: Concatenate the parallel ran results.
  - `visualize/`:
    - `*.py`: generate the figures and tables in paper.

### Raw Experiment Data Format

Raw experiments data is stored in `data/` folder.

- Main Results:
  - In the folder `main_results/[dataset]/`.
  - `resize_aliasing/` and `rotation_aliasing/`  stores the maximize aliasing estimation for scaling and rotation respectively.
    - For scaling, the naming is `[minimum-ratio]_[maximum-ratio]-[N]-[R]`.
    - For rotation, the naming is `[N]-[R]-[angle]`, where angle stands for [-angle, angle] rotation angle interval.
    - In each file, each row records the squared maximum aliasing estimation and running time of the test set sample (indexed from 0).
  - The other folder specifies the name of model structure, inside the folder, each semantic transformation or its composition data is stored in corresponding folder.
    - `noise_*` stores the raw data, i.e., the ceritifed radius and running time for each sample. And the file naming specifies the model training and certifying parameter.
    - `noise_*.report` is the corresponding high-level statistics of robust accuracy.
- Study of Number of Samples for Rotation Transformation
  - In the folder `rotation_sampling_number_study`
  - On CIFAR-10 dataset
  - The number of samples, N and R, are specified in corresponding folders
  - In each folder, the aliasing data is stored in the file named `[N]_[R]_p_10`, where `p_10` stands for certifying +-10 degrees
  - The detail certification data for each model is specified by model trained and certifying noise standard deviation 0.05/0.10/0.15.

### Usage

In this section we demonstrate the training and certifying usage for each transformation by examples.

The detail usage is revealed by the detailed parameter description of each script.

#### Gaussian Blur

##### Training

Example 1: cifar10, using Gaussian blur augmentation with exponential noise (lambda = 2) on GPU 1:

`~/anaconda3/bin/python semantic/train.py cifar10 cifar_resnet110 expgaussian models/cifar10/resnet110/expgaussian/noise_2 --batch 400 --noise_sd 2 --gpu 1`

- `cifar10`: dataset name, can change to `mnist` and `imagenet`
- `cifar_resnet110`: model structure
- `expgaussian`: type of transformation and noise distribution
- `models/***`: model output
- `--batch 400`: batch size
- `--noise_sd 2`: exponential distirbution lambda
- `--gpu 1`: GPU No.

Example 2:  imagenet train Gaussian blur augmentation with exponential noise (lambda = 10) with pretrain model from trochvision resnet-50 on gpu 1 for 3 epochs:

` ~/anaconda3/bin/python semantic/train.py imagenet resnet50 expgaussian models/imagenet/resnet50/expgaussian/pretrain_noise_10 --batch 80 --noise_sd 10 --pretrain torchvision --lr 0.001 --gpu 1 --epochs 3`

##### Certify

Example 1: imagenet, using Gaussian with exponential noise(lambda = 10)

`~/anaconda3/bin/python semantic/certify.py imagenet models/imagenet/resnet50/expgaussian/pretrain_noise_10/checkpoint.pth.tar 10.0 expgaussian data/predict/imagenet/resnet50/expgaussian/pretrain_noise_10 --skip 500 --batch 400 --N 10000`

- `imagenet`: dataset name, can change to `mnist` and `cifar10`
- `models/***`: the trained model path
- `10.0`: exponential distribution lambda
- `expgaussian`: Gaussian blur with exponential distribution
- `data/***`: output certify result
- `--skip 500`: pick one sample in test set for every `500` samples
- `--batch 400`: batch size
- `--N 10000`: randomized smoothing parameter $N$.

Example 2: mnist, using Gaussian with exponential noise(lambda=2)

`~/anaconda3/bin/python semantic/certify.py mnist models/mnist/mnist_43/expgaussian/noise_2/checkpoint.pth.tar 2 expgaussian data/predict/mnist/mnist_43/expgaussian/noist_2 --skip 20 --batch 400`



*We can also use uniform distribution for data augmentation training and certiying.*

##### Train

imagenet: Gaussian blur with uniform noise distribution $U([0, 10^2])$, pretrained from torchvision resnet50 model, learning rate 0.001, on GPU 1

` ~/anaconda3/bin/python semantic/train.py imagenet resnet50 gaussian models/imagenet/resnet50/gaussian/pretrain_noise_10 --batch 80 --noise_sd 10 --pretrain torchvision --lr 0.001 --gpu 1`

##### Certify

cifar10: Gaussian blur with uniform noise distribution $U([0, 5^2])$

`~/anaconda3/bin/python semantic/certify.py cifar10 models/cifar10/resnet110/gaussian/noise_5/checkpoint.pth.tar 5.0 gaussian data/predict/cifar10/resnet110/gaussian/noise_5/test/sigma_5 --skip 20 --batch 400`

#### Translation

##### Training

Example 1: cifar10, reflection-padding, with Gaussian distributed (sigma=10.0) perturbation on both x and y axis, on GPU 2

`~/anaconda3/bin/python semantic/train.py cifar10 cifar_resnet110 translation models/cifar10/resnet110/translation/noise_10.0 --batch 400 --noise 10.0 --gpu 2`

Example 2: mnist, black-padding, with Gaussian distributed (sigma=10.0) perturbation on both x and y axis, on GPU 2

`~/anaconda3/bin/python semantic/train.py mnist mnist_43 btranslation models/mnist/mnist_43/btranslation/noise_10.0 --batch 400 --noise_sd 10.0 --epochs 20 --gpu 2`



##### Certify by Randomized Smoothing

Example 1: imagenet, reflection-padding, with Gaussian distributed (sigma=10.0) perturbation on both x and y axis, on default GPU

`~/anaconda3/bin/python semantic/certify.py imagenet models/imagenet/resnet50/translation/pretrain_noise_10.0/checkpoint.pth.tar 10.0 translation data/predict/imagenet/resnet50/translation/pretrain_noise_10.0 --skip 500 --batch 400 --N 10000`

Example 2: mnist, reflection-padding, with Gaussian distributed (sigma=3.0) perturbation on both x and y axis, on default GPU

`~/anaconda3/bin/python semantic/certify.py mnist models/mnist/mnist_43/translation/noise_3.0/checkpoint.pth.tar 3.0 translation data/predict/mnist/mnist_43/translation/noise_3.0 --skip 20 --batch 400`

##### Certify by Enumeration

Example 1: mnist, reflection-padding

`~/anaconda3/bin/python semantic/translation_certify.py mnist models/mnist/mnist_43/translation/noise_3.0/checkpoint.pth.tar translation data/predict/mnist/mnist_43/single_translation/noise_3.0 --skip 20`

Example 2: imagenet, black-padding

`~/anaconda3/bin/python semantic/translation_certify.py cifar10 models/cifar10/resnet110/btranslation/noise_10.0/checkpoint.pth.tar btranslation data/predict/cifar10/resnet110/single_btranslation/noise_10.0 --skip 20`

#### Brightness and Contrast

##### Training

Example 1: cifar, using Gaussian noise (on contrast dimension sigma_k=0.2, on brightness dimension sigma_b=0.2), on GPU1

`~/anaconda3/bin/python semantic/train.py cifar10 cifar_resnet110 brightness models/cifar10/resnet110/brightness/noise_0.2_0.2 --batch 400 --noise_k 0.2 --noise_b 0.2 --gpu 1`

Example 2: imagenet, using Gaussian noise (on contrast dimension sigma_k=0.2, on brightness dimension sigma_b=0.2), pretrained from torchvision resnet50 model, learning rate 0.001, on GPU1

`~/anaconda3/bin/python semantic/train.py imagenet resnet50 brightness models/imagenet/resnet50/brightness/pretrain_noise_0.2_0.2 --batch 80 --noise_k 0.2 --noise_b 0.2 --batch 80 --pretrain torchvision --lr 0.001 --gpu 1`

##### Certify

1. Brightness Certification

   cifar10, using Gaussian noise (on contrast dimension sigma_k=0.0, on brightness dimension sigma_b=0.2)

   Note that the noise deviations are solely depended on new parameter `noise_k` and `noise_b`, the default noise parameter has no use here.

   The parameter `bright_scale` decides the contrast margin we need for the composition of brightness and contrast. Here since we only certify brightness, we set `bright_scale=0.0`.

   `~/anaconda3/bin/python semantic/certify.py cifar10 models/cifar10/resnet110/brightness/noise_0.2_0.2/checkpoint.pth.tar 0.0 brightness data/predict/cifar10/resnet110/brightness/pretrain_noise_0.2_0.2_by_k_0.0 --noise_k 0.0 --noise_b 0.2 --bright_scale 0.0 --skip 20 --batch 400 --N 100000`

2. Brightness and Contrast Certification

   imagenet, using Gaussian noise (on contrast dimension sigma_k=0.2, on brightness dimension sigma_b=0.2), permitting contrast change +=20%

   `~/anaconda3/bin/python semantic/certify.py imagenet models/imagenet/resnet50/brightness/pretrain_noise_0.2_0.2/checkpoint.pth.tar 0.0 brightness data/predict/imagenet/resnet50/brightness_contrast/pretrain_noise_0.2_0.2_by_k_0.2 --noise_k 0.2 --noise_b 0.2 --bright_scale 0.2 --skip 500 --batch 400 --N 10000`

3. Contrast Certification

- cifar10, using Gaussian noise (on contrast dimension sigma_k=0.2, on brightness dimension sigma_b=0.2)

- `~/anaconda3/bin/python semantic/certify.py cifar10 models/cifar10/resnet110/brightness/noise_0.2_0.2/checkpoint.pth.tar 0.0 contrast data/predict/cifar10/resnet110/contrast/pretrain_noise_0.2_0.2 --noise_k 0.2 --noise_b 0.2 --skip 20 --batch 400 --N 100000`

#### Rotation

##### Compute Aliasing

Example: imagenet, N=10000 R=1000 every 500 sample +-10 degree rotation

`~/anaconda3/bin/python semantic/strict_rotation_aliasing_analyze.py imagenet data/predict/imagenet/rotation_alias_stat/10000_1000_p_10 --skip 500 --slice 10000 --subslice 1000 --partial 10`

##### Training

Example: imagenet, additive Gaussian noise augmentation with std sigma=0.50,  uniform rotation in +-12.5 degree, load pretrain weights from "xxxx/checkpoint.pth.tar", batch size 80, learning rate 0.001, on GPU2

`~/anaconda3/bin/python semantic/train.py imagenet resnet50 strict-rotation-noise models/imagenet/resnet50/rotation/pretrain_noise_0.50_r_12.5 --pretrain xxxx/checkpoint.pth.tar --batch 80 --noise_sd 0.50 --rotation_angle 12.5  --lr 0.001 --gpu 2`

##### Certify

Example: imagenet, additive Gaussian noise augmentation with std sigma=0.50, the aliasing stats data in file `data/predict/imagenet/rotation_alias_stat/10000_1000_p_10`, output results to `data/predict/imagenet/resnet50/rotation/pretrain_noise_0.50_r_12.5`, certify +-10 degree (this setting should be aligned with aliasing stats data), sample 10000 slices (this setting should be aligned with aliasing stats data), randomized smoothing parameter N=10000

`~/anaconda3/bin/python semantic/strict_rotation_certify.py imagenet models/imagenet/resnet50/rotation/pretrain_noise_0.50_r_12.5/checkpoint.pth.tar 0.50 data/predict/imagenet/rotation_alias_stat/10000_1000_p_10 data/predict/imagenet/resnet50/rotation/pretrain_noise_0.50_r_12.5 --N 10000 --skip 500 --batch 400 --slice 10000 --partial 10`

#### Scaling

##### Compute Aliasing

Example: imagenet, N=1000 R=250 every 500 sample +-10% (0.9 - 1.1) scaling

`~/anaconda3/bin/python semantic/strict_resize_aliasing_analyze.py imagenet data/predict/imagenet/resize_alias_stat/0.9_1.1_1000_250 0.9 1.1 --skip 500 --slice 1000 --subslice 250`

##### Training

Example: imagenet, additive Gaussian noise augmentation with std sigma=0.50, uniform scaling in +=15% load pretrain weights from "xxxx/checkpoint.pth.tar", batch size 80, learning rate 0.001, on GPU2

`~/anaconda3/bin/python semantic/train.py imagenet resnet50 resize models/imagenet/resnet50/scaling/pretrain_noise_0.50_p0.15 --pretrain xxxx/checkpoint.pth.tar --batch 80 --noise_sd 0.50 --sl 0.85 --sr 1.15 --lr 0.001 --gpu 2`

##### Certify

Example: imagenet, additive Gaussian noise augmentation with std sigma=0.50, the aliasing stats data in file `data/predict/imagenet/resize_alias_stat/0.9_1.1_1000_250`, output results to `data/predict/imagenet/resnet50/scaling/pretrain_noise_0.50_p0.15`, certify +-10% (0.9-1.1) (this setting should be aligned with aliasing stats data), sample 1000 slices (this setting should be aligned with aliasing stats data), randomized smoothing parameter N=10000

`~/anaconda3/bin/python semantic/strict_resize_certify.py imagenet models/imagenet/resnet50/scaling/pretrain_noise_0.50_p0.15/checkpoint.pth.tar 0.9 1.1 0.50 data/predict/imagenet/resize_alias_stat/0.9_1.1_1000_250 data/predict/imagenet/resnet50/scaling/pretrain_noise_0.50_p0.15 --skip 500 --batch 400 --slice 1000 --N 10000`

#### Data Analysis

The data analysis is done by `semantic/analyze.py`.

Its usage:

`python semantic/analyze.py [logfile] [outfile] [budget=0.0] [step=0.25]`

- logfile: input detail certification result file, which should be the output of certification script
- outfile: output reader-friendly stats file, which output the robust accuracy for radius per step.
- budget: additional budget requirement of robustness. For example, in rotation and scaling we require at least $M$ radius to be judged as robust
- step: robust accuracy computation step size.

Besides to file, the output also goes to stdout, which additionally include the clean accuracy of the model.

