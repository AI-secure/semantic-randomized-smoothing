import pandas as pd
import time
from dateutil import parser

def analyze(csv_path):
    df = pd.read_csv(csv_path, delimiter="\t")
    n = len(df)
    # print(f'Total: {n} records')
    recs = list()
    for i in range(n):
        dt = parser.parse(df.loc[i]['time'])
        recs.append(dt.time().hour * 3600 + dt.time().minute * 60 + dt.time().second + dt.time().microsecond / 1E6)
    return len(recs), sum(recs) / len(recs), min(recs), max(recs)

def nice_show(paths):
    for path in paths:
        n, avg, a, b = analyze(path)
        print(f"{path}:\n  n = {n}\n  avg = {avg:.4}\n  min = {a:.4}\n  max = {b:.4}")

def time_conv(time):
    s = f'${time:.3}$'
    if 'e+' in s:
        s = f'${round(time)}$'
    return s

def table_generator(paths, settings):
    dataset_mapping = {'mnist': 'MNIST', 'cifar10': 'CIFAR-10', 'imagenet': 'ImageNet'}
    trans_mapping = {'expgaussian': 'Gaussian Blur',
                     'brightness': 'Brightness',
                     'contrast': 'Contrast',
                     'brightness_contrast': 'Contrast and Brightness',
                     'translation': 'Translation w/ Reflection-Pad.',
                     'single_translation': 'Translation w/ Reflection-Pad.',
                     'single_btranslation': 'Translation w/ Black-Pad.',
                     'rotation': 'Rotation',
                     'strict-rotation-noise': 'Rotation',
                     'scaling': 'Scaling',
                     'resize': 'Scaling'}
    method_mapping = {'expgaussian': 'Rand. Smooth',
                      'brightness': 'Rand. Smooth',
                      'contrast': 'Rand. Smooth',
                      'brightness_contrast': 'Rand. Smooth',
                      'translation': 'Rand. Smooth',
                      'single_translation': 'Enumeration',
                      'single_btranslation': 'Enumeration',
                      'rotation': 'Sampling Rand. Smooth',
                      'strict-rotation-noise': 'Sampling Rand. Smooth',
                      'scaling': 'Sampling Rand. Smooth',
                      'resize': 'Sampling Rand. Smooth'}

    table = []
    for i,path in enumerate(paths):
        n, avg, a, b = analyze(path)
        fields = path.split('/')
        table.append([dataset_mapping[fields[3]], trans_mapping[fields[5]], '$' + settings[i] + '$', method_mapping[fields[5]],
                      time_conv(avg), time_conv(a), time_conv(b)])
    # f'${avg:.4}$', f'${a:.4}$', f'${b:.4}$'])
    for i in range(len(table)-1, 0, -1):
        for j in range(len(table[i])):
            if j in [0,1]:
                if table[i][j] == table[i-1][j]:
                    table[i][j] = ''
    head = '\\toprule\n Dataset & Transformation & Setting & Method & Avg.(s) & Min.(s) & Max.(s) \\\\\n\\midrule \n'
    body = '\\\\\n'.join([' & '.join(x) for x in table])
    tail = '\\\\\n\\bottomrule\n'
    return head + body + tail


if __name__ == '__main__':
    pathlist = [
        'data/data/predict/mnist/mnist_43/expgaussian/noist_2',
        'data/data/predict/mnist/mnist_43/expgaussian/noist_10',
        'data/data/predict/mnist/mnist_43/brightness/noise_0.3_0.3',
        'data/data/predict/mnist/mnist_43/contrast/noise_0.3_0.3',
        'data/data/predict/mnist/mnist_43/brightness_contrast/noise_0.3_0.3',
        'data/data/predict/mnist/mnist_43/translation/noise_3.0',
        'data/data/predict/mnist/mnist_43/translation/noise_10.0',
        'data/data/predict/mnist/mnist_43/single_translation/noise_3.0',
        'data/data/predict/mnist/mnist_43/single_translation/noise_10.0',
        'data/data/predict/mnist/mnist_43/single_btranslation/noise_3.0',
        'data/data/predict/mnist/mnist_43/single_btranslation/noise_10.0',
        'data/data/predict/mnist/mnist_43/rotation/noise_0.10_r_35_p_30',
        'data/data/predict/mnist/mnist_43/rotation/noise_0.25_r_35_p_30',
        'data/data/predict/mnist/mnist_43/rotation/noise_0.50_r_35_p_30',
        'data/data/predict/mnist/mnist_43/scaling/noise_0.10_p_0.25_t_0.2',
        'data/data/predict/mnist/mnist_43/scaling/noise_0.25_p_0.25_t_0.2',
        'data/data/predict/mnist/mnist_43/scaling/noise_0.50_p_0.25_t_0.2',

        'data/data/predict/cifar10/resnet110/expgaussian/noise_2',
        'data/data/predict/cifar10/resnet110/expgaussian/noise_10',
        'data/data/predict/cifar10/resnet110/expgaussian/noise_50',
        'data/data/predict/cifar10/resnet110/brightness/pretrain_noise_0.2_0.2_by_k_0.0',
        'data/data/predict/cifar10/resnet110/contrast/pretrain_noise_0.2_0.2',
        'data/data/predict/cifar10/resnet110/brightness_contrast/noise_0.2_0.2/test/sigma_0.2_0.2_k_0.2',
        'data/data/predict/cifar10/resnet110/translation/noise_3.0/test/sigma_3.0',
        'data/data/predict/cifar10/resnet110/translation/noise_10.0/test/sigma_10.0',
        'data/data/predict/cifar10/resnet110/single_translation/noise_3.0',
        'data/data/predict/cifar10/resnet110/single_translation/noise_10.0',
        'data/data/predict/cifar10/resnet110/single_btranslation/noise_3.0',
        'data/data/predict/cifar10/resnet110/single_btranslation/noise_10.0',
        'data/data/predict/cifar10/resnet110/strict-rotation-noise/noise_0.00_r_12.5_w_0.05',
        'data/data/predict/cifar10/resnet110/strict-rotation-noise/noise_0.05_r_12.5',
        'data/data/predict/cifar10/resnet110/strict-rotation-noise/noise_0.10_r_12.5',
        'data/data/predict/cifar10/resnet110/strict-rotation-noise/noise_0.15_r_12.5',
        'data/data/predict/cifar10/resnet110/resize/noise_0.05_0.75_1.25_by_0.8_1.2_1000_250',
        'data/data/predict/cifar10/resnet110/resize/noise_0.10_0.75_1.25_by_0.8_1.2_1000_250',
        'data/data/predict/cifar10/resnet110/resize/noise_0.15_0.75_1.25_by_0.8_1.2_1000_250',
        'data/data/predict/cifar10/resnet110/resize/noise_0.20_0.75_1.25_by_0.8_1.2_1000_250',
        'data/data/predict/cifar10/resnet110/resize/noise_0.25_0.75_1.25_by_0.8_1.2_1000_250',
        'data/data/predict/cifar10/resnet110/resize/noise_0.30_0.75_1.25_by_0.8_1.2_1000_250',

        'data/data/predict/imagenet/resnet50/expgaussian/pretrain_noise_10',
        'data/data/predict/imagenet/resnet50/brightness/pretrain_noise_0.2_0.2_by_k_0.0',
        'data/data/predict/imagenet/resnet50/contrast/pretrain_noise_0.2_0.2',
        'data/data/predict/imagenet/resnet50/brightness_contrast/pretrain_noise_0.2_0.2_by_k_0.2',
        'data/data/predict/imagenet/resnet50/translation/pretrain_noise_3.0',
        'data/data/predict/imagenet/resnet50/translation/pretrain_noise_10.0',
        'data/data/predict/imagenet/resnet50/single_translation/noise_3.0',
        'data/data/predict/imagenet/resnet50/single_translation/noise_10.0',
        'data/data/predict/imagenet/resnet50/single_btranslation/noise_3.0',
        'data/data/predict/imagenet/resnet50/single_btranslation/noise_10.0',
        'data/data/predict/imagenet/resnet50/rotation/pretrain_noise_0.25_r_12.5',
        'data/data/predict/imagenet/resnet50/rotation/pretrain_noise_0.50_r_12.5',
        'data/data/predict/imagenet/resnet50/scaling/pretrain_noise_0.50_p0.15'
    ]
    # nice_show(pathlist)

    with open('data/settings.txt', 'r') as f:
        data = f.readlines()
        settings = ['' for _ in pathlist]
        for item in data:
            fields = item.split(' ')
            if fields[1].strip() in pathlist:
                settings[pathlist.index(fields[1].strip())] = fields[0]
    tab = table_generator(pathlist, settings)
    print(tab)
