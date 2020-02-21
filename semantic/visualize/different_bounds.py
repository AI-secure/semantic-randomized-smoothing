import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

out_pdf1 = 'visualize/bound_compare_a.pdf'
out_pdf2 = 'visualize/bound_compare_b.pdf'

if __name__ == '__main__':
    x = np.arange(0.5, 1.0, 0.001)

    y1 = norm.ppf(x)
    y2 = 1.0 * np.sqrt(3) * (2.0 * x - 1)
    y3 = 1.0 * np.sqrt(3) * (2.0 * x - 1)
    y4 = -1.0 / np.sqrt(2) * np.log(2.0 - 2.0 * x)
    y5 = - np.log(2.0 - 2.0 * x)
    y6 = np.sqrt(np.pi / (np.pi - 2.0)) * norm.ppf(0.5 + x / 2.0) - np.sqrt(np.pi / (np.pi - 2.0)) * norm.ppf(0.75)


    # y1 = np.sqrt(np.pi / 2.0) * norm.ppf(x)
    # y2 = 2.0 * (2.0 * x - 1)
    # y3 = 1.0 * (2.0 * x - 1)
    # y4 = - np.log(2.0 - 2.0 * x)
    # y5 = - np.log(2.0 - 2.0 * x)
    # y6 = np.sqrt(np.pi / 2.0) * (np.sqrt(np.pi / (np.pi - 2.0)) * norm.ppf(0.5 + x / 2.0) - np.sqrt(np.pi / (np.pi - 2.0)) * norm.ppf(0.75))

    fig, ax = plt.subplots()
    ax.set(xlabel='$p_A$', ylabel='Robust Radius for $\delta$')

    ax.plot(x, y1, 'g-', label='iid Gaussian')
    ax.plot(x, y2, 'r-', label='Uniform')
    ax.plot(x, y4, 'm-', label='Laplace')

    legend = ax.legend(loc='upper left', shadow=True, fontsize='large')

    plt.title('Robust Radius Comparison for Different Noise Distributions\n($\sigma^2=1,m=1$)')

    # plt.show()
    plt.savefig(out_pdf1)

    ax.cla()
    ax.plot(x, y1, 'g-', label='iid Gaussian')
    ax.plot(x, y2, 'r-', label='Uniform')
    ax.plot(x, y5, 'y--', label='Exponential')
    ax.plot(x, y6, 'g--', label='Folded Gaussian')

    legend = ax.legend(loc='upper left', shadow=True, fontsize='large')

    plt.title('Robust Radius Comparison for Different Noise Distributions\n($\sigma^2=1,m=1$)')

    # plt.show()
    plt.savefig(out_pdf2)

