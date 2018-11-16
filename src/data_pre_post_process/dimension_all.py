import matplotlib.pyplot as plt
from matplotlib import ticker


font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20, }


if __name__ == '__main__':
    x = [4, 8, 16, 32, 64]
    y1 = [1.668 * 0.001, 1.477 * 0.001, 1.189 * 0.001, 0.994 * 0.001, 0.965 * 0.001]
    y2 = [1.312 * 0.001, 1.248 * 0.001, 1.189 * 0.001, 1.189 * 0.001, 1.189 * 0.001]
    y3 = [0.0, 0.0, 0.0, 0.0, 0.0]

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y1, color='r', marker='s',
        linewidth=2, markersize=10, markerfacecolor='none')
    ax.set_xlabel("Embedding Dimension", fontsize=20)
    ax.set_ylabel("mse", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(linestyle='--')
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0.002))
    ax.yaxis.set_major_formatter(formatter)
    plt.axis([0, 70, 0, 0.002])
    plt.savefig('embedding.eps', bbox_inches='tight', format='eps', dpi=1000)
    plt.savefig('embedding.png', bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y2, color='b', marker='s',
            linewidth=2, markersize=10, markerfacecolor='none')
    ax.set_xlabel("# Histogram Bins", fontsize=20)
    ax.set_ylabel("mse", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(linestyle='--')
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0.002))
    ax.yaxis.set_major_formatter(formatter)
    plt.axis([0, 70, 0, 0.002])
    plt.savefig('Histogram Bins.eps', bbox_inches='tight')
    plt.savefig('Histogram Bins.png', bbox_inches='tight')
    # plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y3, color='g', marker='s',
            linewidth=2, markersize=8, markerfacecolor='none')
    ax.set_xlabel("# NTN feature maps", fontsize=20)
    ax.set_ylabel("mse", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(linestyle='--')
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0.002))
    ax.yaxis.set_major_formatter(formatter)
    plt.axis([0, 70, 0, 0.002])
    plt.savefig('NTN.png')
    # plt.show()

