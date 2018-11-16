import networkx as nx
import numpy as np
import random
from config import FLAGS
from distance import normalized_ged
# from distance import ged
# from utils import exec_turnoff_print, exec_turnon_print


def graph_generator(nxgraph, sample_num):
    sample_graphs, sample_geds, num = [], [], 0
    # iter_max = 10
    while num < sample_num:
        g_candi, ged_candi = mask_generator(nxgraph.copy())
        sample_graphs.append(g_candi)
        if FLAGS.dist_norm:
            ged_candi = normalized_ged(ged_candi, nxgraph, g_candi)
        sample_geds.append(ged_candi)
        num += 1
        # iter_max -= 1
    return sample_graphs, sample_geds


def add_node_edge(graph, nodes, idx):
    nodes = [int(n) for n in nodes]
    new_node = str(max(nodes) + 1)
    graph.add_edge(str(nodes[idx]), new_node)
    if FLAGS.node_feat_name is not None:
        nx.set_node_attributes(graph, FLAGS.node_feat_name, {new_node: 'C'})  # TODO: multiple element
    assert (len(graph.nodes()) == len(nodes) + 1)
    return graph


def mask_generator(graph):
    adj = np.array(nx.to_numpy_matrix(graph))
    nodes = graph.nodes()
    edges = graph.edges()
    graph_copy = graph.copy()
    if len(nodes) < 2:
        raise RuntimeError('Wrong graph {} with {} nodes'.format(
            graph.graph['gid'], graph.number_of_nodes()))
    if len(nx.isolates(graph)) != 0:
        raise RuntimeError('Wrong graph {} with {} isolated nodes'.format(
            graph.graph['gid'], len(nx.isolates(graph))))
    if len(nodes) > FLAGS.max_nodes:
        raise RuntimeError('Wrong graph {} with {} nodes, more than max nodes'.format(
            graph.graph['gid'], graph.number_of_nodes()))
    # assert (len(nodes) >= 2 and len(nx.isolates(graph)) == 0 and len(nodes) > FLAGS.max_nodes)

    row, col = random.sample(range(0, len(graph.nodes())), 2)

    # Add edge / Add node and edge
    if adj[row][col] == 0:
        assert (adj[col][row] == 0)
        if len(nodes) == FLAGS.max_nodes or random.randint(0, 1):
            graph_copy.add_edge(nodes[row], nodes[col])
            assert (len(graph_copy.edges()) == len(edges) + 1)
            ged_candi = 1
        else:
            graph_copy = add_node_edge(graph_copy, nodes, row)
            ged_candi = 2
        g_candi = graph_copy

    # Delete edge / Delete edge & node
    elif adj[row][col] == 1:
        assert (adj[col][row] == 1)
        graph_copy.remove_edge(nodes[row], nodes[col])
        assert (len(graph_copy.edges()) == len(edges) - 1)
        if not nx.is_connected(graph_copy):
            iso_num = len(nx.isolates(graph_copy))
            if iso_num == 0:
                graph_copy = add_node_edge(graph.copy(), nodes, row)
                ged_candi = 2
            elif iso_num == 1:
                graph_copy.remove_node(nx.isolates(graph_copy)[0])
                assert (len(graph_copy.nodes()) == len(nodes) - 1)
                ged_candi = 2
            elif iso_num == 2:
                graph_copy = graph
                ged_candi = 0
            else:
                raise RuntimeError('Wrong graph: {} more than 2 isolated node'.format(
                    graph.graph['gid']))
        else:
            ged_candi = 1
        g_candi = graph_copy

    else:
        raise Exception

    # exec_turnoff_print()
    # if FLAGS.dataset == 'imdbmulti':
    #     algo = 'beam80'
    # else:
    #     algo = FLAGS.dist_algo
    # ged_candi = ged(g_candi, graph, algo)
    # exec_turnon_print()
    return g_candi, ged_candi

#
# # Toy case
# G = nx.Graph()
# G.add_edge(1, 2)
# G.add_edge(1, 3)
# G.add_edge(1, 4)
# G.add_edge(2, 3)
# G.add_edge(2, 4)
# print(G.edges())
# print(nx.to_numpy_matrix(G))
# print('-------------------')
# for _ in range(10):
#     H = G.copy()
#     H_new, ged_new = graph_generator(H, 5, 3)
#     for i in range(len(H_new)):
#         print(ged_new[i])
#         print(H_new[i].edges())
#         print(nx.to_numpy_matrix(H_new[i]))
#     print('=================')
