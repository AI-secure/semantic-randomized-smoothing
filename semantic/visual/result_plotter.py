import matplotlib.pyplot as plt
import pandas as pd


def plotter(title, x_name, y_name, series_data, series_color, series_name, x_range=None, y_range=None):
    fig, ax = plt.subplots()
    for i in range(len(series_data)):
       ax.plot(series_data[i][0], series_data[i][1], series_color[i], label=series_name[i])
    ax.legend()
    ax.grid()
    ax.set(xlabel=x_name, ylabel=y_name, title=title)
    if x_range is not None:
        plt.xlim(x_range)
    if y_range is not None:
        plt.ylim(y_range)
    return fig

def robacc_reader(csv_path):
    df = pd.read_csv(csv_path, delimiter="\t")
    print(f'Total: {len(df)} records')
    rad = list()
    n = len(df)
    for i in range(n):
        if df.loc[i]["correct"]:
            rad.append(df.loc[i]["radius"])
    rad = sorted(rad)
    m = len(rad)
    x, y = [0.], [m/n*100.0]
    for i,item in enumerate(rad):
        x.append(item),
        y.append((m-i-1)/n*100.0)
    return x, y

if __name__ == '__main__':
    plt.rcParams.update({'font.size': 10})
    plt.rcParams.update({'figure.autolayout': True})

    # noise compare

    m_x = [0.,1.,2.,3.,4.,5.,6.,7.,8.]

    m1_exp = [88.2, 84.6, 76.6, 58.4, 0.0, 0.0, 0.0, 0.0, 0.0]
    m1_u = [83.2, 80.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    m1_n = [81.4, 77.6, 69.6, 47.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    m2_exp = [76.4, 74.2, 70.0, 62.8, 55.4, 47.8, 40.8, 31.2, 20.2]
    m2_u = [66.8, 65.2, 60.4, 55.8, 45.6, 0.0, 0.0, 0.0, 0.0]
    m2_n = [69.8, 68.2, 64.2, 57.4, 50.6, 42.4, 35.4, 0.0, 0.0]

    m3_exp = [70.4, 69.8, 67.0, 63.4, 57.0, 49.4, 43.2, 37.4, 32.6]
    m3_u = [57.0, 56.8, 54.8, 51.0, 47.8, 40.8, 0.0, 0.0, 0.0]
    m3_n = [64.8, 63.6, 61.0, 56.6, 50.4, 43.6, 37.0, 32.4, 27.4]

    m_fig = plotter('Robust Accuracy for Guassian Blur\n'
                    'Using Different Noise Distributons on CIFAR-10\n'
                    '$\sigma_{train}^2=100.0$',
                    x_name='Robust Radii $\sqrt{\\alpha}$',
                    y_name='Robust Accuracy(%)',
                    series_data=[[m_x, m2_exp], [m_x, m2_u], [m_x, m2_n]],
                    series_color=['ro-','g+:','bv--'],
                    series_name=['Exponential Dist.', 'Uniform Dist.', 'Normal Dist.'])
    m_fig.set_size_inches(640/96, 480/96, forward=True)
    m_fig.savefig('visualize/noise_compare_gaussian_blur_cifar10_100.pdf')
    m_fig.show()

    # gaussian blur

    g_x = [0.,1.,2.,3.,4.,5.,6.,7.,8.]

    g_2 = [88.2, 84.6, 76.6, 58.4, 0.0, 0.0, 0.0, 0.0, 0.0]
    g_10 = [76.4, 74.2, 70.0, 62.8, 55.4, 47.8, 40.8, 31.2, 20.2]
    g_20 = [70.4, 69.8, 67.0, 63.4, 57.0, 49.4, 43.2, 37.4, 32.6]

    m_fig = plotter('Robust Accuracy for Gaussian Blur\n'
                    'Using Different Exponential Noise Variance on CIFAR-10\n',
                    x_name='Robust Radii $\sqrt{\\alpha}$',
                    y_name='Robust Accuracy(%)',
                    series_data=[[g_x, g_2], [g_x, g_10], [g_x, g_20]],
                    series_color=['ro-','g+:','bv--'],
                    series_name=['$\lambda=2.0$','$\lambda=10.0$','$\lambda=20.0$'])
    m_fig.set_size_inches(640 / 96, 480 / 96, forward=True)
    m_fig.savefig('visualize/noise_compare_gaussian_blur_cifar10_var.pdf')
    m_fig.show()

    # random smooth vs. enumeration

    m_10_x_r, m_10_y_r = robacc_reader('data/data/predict/mnist/mnist_43/translation/noise_10.0')
    m_10_x_e, m_10_y_e = robacc_reader('data/data/predict/mnist/mnist_43/single_translation/noise_10.0')
    c_10_x_r, c_10_y_r = robacc_reader('data/data/predict/cifar10/resnet110/translation/noise_10.0/test/sigma_10.0')
    c_10_x_e, c_10_y_e = robacc_reader('data/data/predict/cifar10/resnet110/single_translation/noise_10.0')
    i_10_x_r, i_10_y_r = robacc_reader('data/data/predict/imagenet/resnet50/translation/pretrain_noise_10.0')
    i_10_x_e, i_10_y_e = robacc_reader('data/data/predict/imagenet/resnet50/single_translation/noise_10.0')

    m_fig = plotter('Comparison of Randomized Smoothing and Enumeration\n'
                    'for Translation with Reflection-Padding',
                    x_name='$\sqrt{\Delta x^2 + \Delta y^2}$',
                    y_name='Robust Accuracy(%)',
                    series_data=[[m_10_x_r, m_10_y_r], [m_10_x_e, m_10_y_e],
                                 [c_10_x_r, c_10_y_r], [c_10_x_e, c_10_y_e],
                                 [i_10_x_r, i_10_y_r], [i_10_x_e, i_10_y_e]],
                    series_color=['r-', 'r:', 'g-', 'g:', 'b-', 'b:'],
                    series_name=['MNIST $\sigma=10$ Rand. Smooth', 'MNIST $\sigma=10$ Enumeration',
                                 'CIFAR-10 $\sigma=10$ Rand. Smooth', 'CIFAR-10 $\sigma=10$ Enumeration',
                                 'ImageNet $\sigma=10$ Rand. Smooth', 'ImageNet $\sigma=10$ Enumeration'],
                    x_range=(0, 25))
    m_fig.savefig('visualize/translation_rand_smooth_enumeration_comp.pdf')
    m_fig.show()
    
    # reflection-padding vs. black padding

    # m_10_x_r, m_10_y_r = robacc_reader('data/data/predict/mnist/mnist_43/single_translation/noise_10.0')
    m_10_x_b, m_10_y_b = robacc_reader('data/data/predict/mnist/mnist_43/single_btranslation/noise_10.0')
    # c_10_x_r, c_10_y_r = robacc_reader('data/data/predict/cifar10/resnet110/single_translation/noise_10.0')
    c_10_x_b, c_10_y_b = robacc_reader('data/data/predict/cifar10/resnet110/single_btranslation/noise_10.0')
    # i_10_x_r, i_10_y_r = robacc_reader('data/data/predict/imagenet/resnet50/single_translation/noise_10.0')
    i_10_x_b, i_10_y_b = robacc_reader('data/data/predict/imagenet/resnet50/single_btranslation/noise_10.0')

    m_fig = plotter('Comparision of Translation with Reflection-Padding and Black-Padding',
                    x_name='$\sqrt{\Delta x^2 + \Delta y^2}$',
                    y_name='Robust Accuracy(%)',
                    series_data=[[m_10_x_r, m_10_y_r], [m_10_x_b, m_10_y_b],
                                 [c_10_x_r, c_10_y_r], [c_10_x_b, c_10_y_b],
                                 [i_10_x_r, i_10_y_r], [i_10_x_b, i_10_y_b]],
                    series_color=['r-', 'r:', 'g-', 'g:', 'b-', 'b:'],
                    series_name=['MNIST $\sigma=10$ Refl.-Pad.',
                                 'MNIST $\sigma=10$ Black-Pad.',
                                 'CIFAR-10 $\sigma=10$ Refl.-Pad.',
                                 'CIFAR-10 $\sigma=10$ Black-Pad.',
                                 'ImageNet $\sigma=10$ Refl.-Pad.',
                                 'ImageNet $\sigma=10$ Black-Pad.'],
                    x_range=(0, 25))
    m_fig.savefig('visualize/translation_reflect_black_comp.pdf')
    m_fig.show()

