import numpy as np
import scipy.io as sio
import sys
sys.path.append('../')
from utils import load_data, load_as_dict
from results import load_result


if __name__ == '__main__':
    # dataset = 'aids700nef'
    # model = 'astar'
    # data = load_as_dict("/home/songbian/Documents/fork/GraphEmbedding/"
    #                     "data/2018-08-03T01:36:37_test_info.pickle")
    # pred_r = load_result(
    #     dataset, 'siamese', sim_mat=data['sim_mat'], time_mat=data['time_li'])
    #
    # train_data = load_data(dataset, train=True)
    # test_data = load_data(dataset, train=False)
    # row_graphs = test_data.graphs
    # col_graphs = train_data.graphs
    # r = load_result(dataset, model, row_graphs=row_graphs, col_graphs=col_graphs)
    matrix_acyclic = np.loadtxt("/home/songbian/Documents/AAAI/dist_matrix_alkane.txt")
    matrix_alkane = np.loadtxt("/home/songbian/Documents/AAAI/dist_matrix_acyclic.txt")
    print(type(matrix_acyclic))
    print(type(matrix_alkane))
    np.save('acyclic_GED_matrix.npy', matrix_acyclic)
    np.save('alkane_GED_matrix.npy', matrix_alkane)
    # print(data['test_results'])
    # print(data['embs'][330])
    # print(data['embs'][25])
    # print(data['embs'][41])
    # print(data['embs'][169])
    # print(data['embs'][475])
    # print(data['embs'][12])
    # print(data['embs'][357])




