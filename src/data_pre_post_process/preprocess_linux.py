import sys

sys.path.append('..')
from utils import get_root_path, get_data_path, exec, get_file_base_id, \
    load_data, \
    save_as_dict, load_as_dict, get_save_path, create_dir_if_not_exists
import networkx as nx
from glob import glob
from collections import defaultdict
import random
from random import sample, shuffle

random.seed(123)


def gen_graphs():
    dirin = get_data_path()
    file = dirin + '/linux_Format-2'
    # train_dirout = dirin + '/train'
    # test_dirout = dirin + '/test'
    # dirin = get_data_path() + '/iGraph20/datasets'
    # file = dirin + '/nasa.igraph'
    train_dirout = dirin + '/train'
    test_dirout = dirin + '/test'
    graphs = {}
    gid = None
    types_count = defaultdict(int)
    total_num_nodes = 0
    disconnects = set()
    less_than_eq_10 = set()
    types_count_less_than_eq_10 = defaultdict(int)
    total_num_nodes_less_than_eq_10 = 0
    with open(file) as f:
        for line in f:
            ls = line.rstrip().split()
            if ls[0] == 't':
                assert (len(ls) == 3)
                assert (ls[1] == '#')
                if gid:
                    assert (gid not in graphs)
                    graphs[gid] = g
                    print(gid, g.number_of_nodes())
                    if g.number_of_nodes() <= 10 and nx.is_connected(g):
                        less_than_eq_10.add(gid)
                        total_num_nodes_less_than_eq_10 += g.number_of_nodes()
                        d = nx.get_node_attributes(g, 'type')
                        for _, type in d.items():
                            types_count_less_than_eq_10[type] += 1
                    if not nx.is_connected(g):
                        disconnects.add(g)
                g = nx.Graph()
                gid = int(ls[2])
            elif ls[0] == 'v':
                assert (len(ls) == 3)
                type = int(ls[2])
                types_count[type] += 1
                g.add_node(int(ls[1]), type=type)
                total_num_nodes += 1
            elif ls[0] == 'e':
                assert (len(ls) == 4)
                edge_type = int(ls[3])
                assert (edge_type == 0)
                g.add_edge(int(ls[1]), int(ls[2]))
    print(len(graphs), 'graphs in total')
    print(len(types_count), 'node types out of total', total_num_nodes, 'nodes')
    print(len(disconnects), 'disconnected graphs')
    for i in range(10):
        print(i, types_count[i])
    print(len(less_than_eq_10), 'small graphs (<= 10 nodes)')
    print(len(types_count_less_than_eq_10), 'node types out of total',
          total_num_nodes_less_than_eq_10, 'nodes')
    select_dump_graphs(graphs, sorted(list(less_than_eq_10)))


def select_dump_graphs(graphs, less_than_eq_10_ids):
    remove_type(graphs)
    random.Random(123).shuffle(less_than_eq_10_ids)
    tr = 800
    te = 200
    assert (tr + te <= len(graphs))
    dirout = get_data_path() + '/linux'
    train_dir = dirout + '/train'
    create_dir_if_not_exists(train_dir)
    for i in range(tr):
        gid = less_than_eq_10_ids[i]
        nx.write_gexf(graphs[gid], train_dir + '/{}.gexf'.format(gid))
    test_dir = dirout + '/test'
    create_dir_if_not_exists(test_dir)
    for i in range(tr, tr + te):
        gid = less_than_eq_10_ids[i]
        nx.write_gexf(graphs[gid], test_dir + '/{}.gexf'.format(gid))


def remove_type(graphs):
    for g in graphs.values():
        for n in g.nodes():
            del g.node[n]['type']


gen_graphs()
