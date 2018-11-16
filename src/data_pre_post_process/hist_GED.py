import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')
from utils import load_as_dict, load_data
from results import load_result
from dist_calculator import get_train_train_dist_mat, DistCalculator, get_gs_dist_mat


font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18, }


if __name__ == '__main__':
    model = 'astar'
    datasets = ['aids700nef', 'linux', 'imdbmulti']
    plt.figure(figsize=(10, 6))
    for dataset in datasets:
        train_data = load_data(dataset, train=True)
        test_data = load_data(dataset, train=False)
        gs_train = train_data.graphs
        gs_test = test_data.graphs
        dist_calculator = DistCalculator(dataset, 'ged', 'astar')
        matrix = get_gs_dist_mat(gs_train, gs_train, dist_calculator, 'train', 'train', dataset,
                                 'ged', 'astar', False)
        # matrix = get_train_train_dist_mat(dataset, 'ged', 'astar', False)
        print(len(matrix))
        print(len(matrix[0]))
        # train_data = load_data(dataset, train=True)
        # test_data = load_data(dataset, train=False)
        # row_graphs = test_data.graphs
        # col_graphs = train_data.graphs
        # r = load_result(dataset, model, row_graphs=row_graphs, col_graphs=col_graphs)
        data = []
        for i in range(len(matrix)):
            for j in range(i, len(matrix[i])):
                data.append(int(matrix[i][j]))
        max_data = max(data)
        data_size = list(range(0, max_data + 1))
        data_freq = [0] * (max_data + 1)

        for num in data:
            data_freq[num] = data_freq[num] + 1

        if dataset == 'aids700nef':
            plt.plot(data_size, data_freq, color='red', marker='o',
                     markerfacecolor='white', label="AIDS",
                     linewidth=2,
                     markersize=10)
        elif dataset == 'linux':
            plt.plot(data_size, data_freq, color='blue', marker='x',
                     markerfacecolor='white', label="LINUX",
                     linewidth=2,
                     markersize=10)
        else:
            plt.plot(data_size, data_freq, color='green', marker='*',
                     markerfacecolor='white', label="IMDB",
                     linewidth=2,
                     markersize=10)
    plt.xlabel("GED", font2)
    plt.ylabel("#pairs", font2)
    plt.xscale('symlog')
    # plt.title('Train')
    plt.legend(loc='best', prop={'size': 20})
    plt.grid(which=u'both', axis=u'x', linestyle='--', color='grey')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('GED_line_chart.png')
    plt.savefig('GED_line_chart.eps')
    plt.show()

    print('Yes')

