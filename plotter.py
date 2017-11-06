import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_scores(scores, filename):
    f = plt.figure()
    ax = f.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(scores, 'o-', label='Score')
    plt.legend()
    f.savefig('report/plots/{}.pdf'.format(filename), bbox_inches='tight')
