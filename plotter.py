import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import spline
from matplotlib import rc

from csv_utils import read_from_csv

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def plot_scores(scores, filename):
    x_points = [i for i in range(1, len(scores) + 1)]
    y_points = scores

    f = plt.figure()
    ax = f.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    x_smooth = np.linspace(1, len(scores), 200)
    y_smooth = spline(x_points, y_points, x_smooth)

    plt.plot(x_smooth, y_smooth, 'C0', label='Score')
    plt.xlabel('Number of episodes')
    plt.ylabel('Score')
    plt.legend()
    f.savefig('report/plots/{}.pdf'.format(filename), bbox_inches='tight')


def plot_from_csv(filename):
    scores = read_from_csv(filename)
    plot_scores(scores, filename)


if __name__ == '__main__':
    plot_from_csv('hamming')
    plot_from_csv('manhattan')
    plot_from_csv('no_generalization')
    plot_from_csv('same_result')
