import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, integrate
import sys
sys.path.append('../')
from dist_calculator import get_gs_dist_mat, DistCalculator
from utils import load_as_dict, load_data
from results import load_result


if __name__ == '__main__':
    dataset = 'aids700nef'
    dist_metric = 'ged'
    dist_algo = 'astar'
    emb_data = load_as_dict("/home/songbian/Documents/fork/GraphEmbedding/model/Siamese/logs/" \
                  "siamese_regression_aids700nef_2018-08-01T11:52:11(cur_best)/test_info.pickle")
    train_data = load_data(dataset, train=True)
    test_data = load_data(dataset, train=False)
    row_graphs = test_data.graphs
    col_graphs = train_data.graphs
    matrix = load_result(dataset, 'astar', row_graphs=row_graphs, col_graphs=col_graphs)
    pred_r = load_result(
        dataset, 'siamese', sim_mat=emb_data['sim_mat'], time_mat=emb_data['time_li'])
    ids = matrix.sort_id_mat_
    print(len(matrix.dist_norm_mat_))
    print(len(matrix.dist_norm_mat_[0]))
    print(matrix.dist_norm_mat_)
    for i in range(len(row_graphs)):
        q = test_data.graphs[i]
        # gids = np.concatenate([ids[i][:5], [ids[i][int(len(col_graphs) / 2)]], ids[i][-1:]])
        gids = ids[i][:10]
        gs = [matrix.dist_norm_mat_[i][j] for j in gids]
        sns_plot = sns.distplot(gs, kde=False, fit=stats.gamma)
        plt.figure()
        fig = sns_plot.get_figure()
        fig.savefig("Queryhist/hist" + str(i) + ".png")
        # plt.show()


