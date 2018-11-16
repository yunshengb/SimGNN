from sklearn.manifold import TSNE
from utils import get_result_path, create_dir_if_not_exists
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('font', **{'family': 'serif', 'size': 22})
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')
from utils import load_data, load_as_dict
from results import load_result


TRUE_MODEL = 'astar'


def visualize_embeddings(dataset, orig_embs, true_result,
                         thresh_pos, thresh_neg, thresh_pos_sim, thresh_neg_sim,
                         norm, pred_r, eps_dir=None):
    # label_mat, _, _ = true_result.classification_mat(
    #    thresh_pos, thresh_neg, thresh_pos_sim, thresh_neg_sim, norm)
    tsne = TSNE(n_components=2)
    embs = tsne.fit_transform(orig_embs)
    dir = '{}/{}/emb_vis'.format(get_result_path(), dataset)
    create_dir_if_not_exists(dir)
    if eps_dir:
        create_dir_if_not_exists(eps_dir)
    m = np.shape(pred_r.sort_id_mat_)[0]
    n = np.shape(pred_r.sort_id_mat_)[1]
    # m = np.shape(label_mat)[0]
    # n = np.shape(label_mat)[1]
    plt_cnt = 0
    print('TSNE embeddings: {} --> {} to plot'.format(
        orig_embs.shape, embs.shape))
    for i in range(m):
        axis_x_red = []
        axis_y_red = []
        axis_x_blue = []
        axis_y_blue = []
        axis_x_query = []
        axis_y_query = []
        for j in range(10):
            axis_x_blue.append(embs[pred_r.sort_id_mat_[i][j], 0])
            axis_y_blue.append(embs[pred_r.sort_id_mat_[i][j], 1])
        for j in range(n - 10):
            axis_x_red.append(embs[pred_r.sort_id_mat_[i][j + 10], 0])
            axis_y_red.append(embs[pred_r.sort_id_mat_[i][j + 10], 1])
        axis_x_query.append(embs[i + n, 0])
        axis_y_query.append(embs[i + n, 1])

        cm = plt.cm.get_cmap("Reds")

        plt.figure()
        plt.scatter(axis_x_red, axis_y_red, s=30, c=sorted(range(n-10), reverse=False),
                    marker='o', alpha=0.6, cmap=plt.cm.get_cmap("Blues"))
        plt.scatter(axis_x_blue, axis_y_blue, s=15, c=sorted(range(10), reverse=True),
                    marker='s', alpha=0.6, cmap=plt.cm.get_cmap("Reds"))
        plt.scatter(axis_x_query, axis_y_query, s=400, c='limegreen', marker='P', alpha=0.6)
        plt.axis('off')
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_visible(False)
        cur_axes.axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(dir + '/' + str(i) + '.png', bbox_inches='tight', pad_inches=0)
        if eps_dir:
            plt.savefig(eps_dir + '/' + str(i) + '.png',
                        bbox_inches='tight', pad_inches=0)
            plt.savefig(eps_dir + '/' + str(i) + '.eps',
                        bbox_inches='tight', pad_inches=0)
        plt_cnt += 1
        plt.close()
    print('Saved {} embedding visualization plots'.format(plt_cnt))


if __name__ == '__main__':
    data = load_as_dict("/home/songbian/Documents/fork/"
                               "GraphEmbedding/data/"
                               "regression_linux_test_info.pickle")
    embs = data['embs']
    dataset = 'linux'
    thresh_pos = 0.58
    thresh_neg = 0.58
    thresh_pos_sim = 0.5
    thresh_neg_sim = 0.5
    norm = True
    row_graphs = load_data(dataset, train=False).graphs
    col_graphs = load_data(dataset, train=True).graphs
    true_result = load_result(
        dataset, TRUE_MODEL, row_graphs=row_graphs, col_graphs=col_graphs)
    pred_r = load_result(
        dataset, 'siamese', sim_mat=data['sim_mat'], time_mat=data['time_li'])
    visualize_embeddings(dataset, embs, true_result, thresh_pos, thresh_neg,
                         thresh_pos_sim, thresh_neg_sim, norm, pred_r)

