import sys

sys.path.append('..')
from utils import get_data_path, get_file_base_id, create_dir_if_not_exists
import networkx as nx
from glob import glob
import random

random.seed(123)


def gen_imdb_multi():
    dirin = get_data_path() + '/imdb_comedy_romance_scifi/graph'
    k = float('inf')
    lesseqk = []
    for file in glob(dirin + '/*.gexf'):
        g = nx.read_gexf(file)
        gid = get_file_base_id(file)
        print(gid, g.number_of_nodes())
        if g.number_of_nodes() <= k:
            g.graph['gid'] = gid
            for node in g.nodes():
                del g.node[node]['node_class']
            for edge in g.edges_iter(data=True):
                del edge[2]['weight']
            lesseqk.append(g)
    print(len(lesseqk))
    gen_dataset(lesseqk)


def gen_dataset(graphs):
    random.Random(123).shuffle(graphs)
    dirout_train = get_data_path() + '/IMDBMulti/train'
    dirout_test = get_data_path() + '/IMDBMulti/test'
    create_dir_if_not_exists(dirout_train)
    create_dir_if_not_exists(dirout_test)
    for g in graphs[0:1200]:
        nx.write_gexf(g, dirout_train + '/{}.gexf'.format(g.graph['gid']))
    for g in graphs[1200:]:
        nx.write_gexf(g, dirout_test + '/{}.gexf'.format(g.graph['gid']))


gen_imdb_multi()
