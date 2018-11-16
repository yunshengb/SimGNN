import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy import stats, integrate
from utils import load_as_dict, load_data
from results import load_result


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


if __name__ == '__main__':
    dataset = 'aids80nef'
    train_data = load_data(dataset, train=True)
    test_data = load_data(dataset, train=False)
    row_graphs = test_data.graphs
    col_graphs = train_data.graphs
    load_res = load_result(dataset, 'astar', row_graphs=row_graphs, col_graphs=col_graphs)
    data_origin = load_as_dict("/home/songbian/Documents/fork/"
                        "GraphEmbedding/data/"
                        "regression_aids80nef_test_info.pickle")
    data = data_origin['node_embs_list']
    for i in range(len(data)):
        for j in range(len(data[i])):
            if len(data[i]) < 10:
                data[i] = np.pad(data[i], ((0, 10 - len(data[i])),
                                (0, 0)), 'constant', constant_values=(0, 0))

    ids = load_res.sort_id_mat_
    for i in range(len(row_graphs)):
        q = test_data.graphs[i]
        gids = np.concatenate([ids[i][:10], ids[i][-10:]])
        for j in gids:
            result = np.dot(data[i], data[j].T)
            sns_plot = sns.heatmap(result)
            plt.figure()
            fig = sns_plot.get_figure()
            fig.savefig("Heatmap/heatmap" + str(i) + "_" + str(j) + ".png")
            result_array = []
            for m in range(len(result)):
                for n in range(len(result)):
                    result_array.append(result[m][n])
            sns_plot = sns.distplot(result_array, kde=False, fit=stats.gamma)
            plt.figure()
            fig = sns_plot.get_figure()
            fig.savefig("Histgram/histgram" + str(i) + "_" + str(j) + ".png")

    # result = np.dot(data[0], data[1].T)
    # print(len(result))
    # print(len(result[0]))
    # plt.figure()
    # ax = sns.heatmap(result)
    # plt.show()
    # # plt.close()
    # result_array = []
    # for i in range(len(result)):
    #     for j in range(len(result)):
    #         result_array.append(result[i][j])
    # plt.figure()
    # sns_plot = sns.distplot(result_array, kde=False, fit=stats.gamma)
    # plt.show()
