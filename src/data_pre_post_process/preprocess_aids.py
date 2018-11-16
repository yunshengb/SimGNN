import sys

sys.path.append('..')
from utils import get_root_path, exec, get_file_base_id, load_data, \
    save_as_dict, load_as_dict, get_save_path
import networkx as nx
from glob import glob
from collections import defaultdict
import random

random.seed(123)


def gen_aids():
    dirin = get_root_path() + '/data'
    file = dirin + '/AIDO99SD'
    dirout = dirin + '/AIDS'
    g = nx.Graph()
    line_i = 0
    gid = 0
    aids_old_ids = get_old_aids_id()
    charges = set()
    charge_gids = set()
    with open(file) as f:
        for line in f:
            if '$$$$' in line:
                print(gid)
                nx.write_gexf(g, dirout + '/{}.gexf'.format(gid))
                g = nx.Graph(gid=gid)
                line_i = 0
                gid += 1
            else:
                if gid == 40650:
                    print(line, end='')
                ls = line.rstrip().split()
                if len(ls) == 9:
                    nid = line_i - 4
                    ls = line.rstrip().split()
                    type = ls[3]
                    charge = int(ls[6])
                    # if charge != 0:
                    #     print(gid)
                    #     if (gid + 1) in aids_old_ids:
                    #         print('#####'*20)
                    if charge != 0:
                        charges.add((type, charge))
                        charge_gids.add(gid)
                    if charge == 1:
                        charge = 3
                    elif charge == 2:
                        charge = 2
                    elif charge == 3:
                        charge = 1
                    elif charge == 4:
                        raise RuntimeError('Cannot handle doublet radical')
                    elif charge == 5:
                        charge = -1
                    elif charge == 6:
                        charge = -2
                    elif charge == 7:
                        charge = -3
                    elif charge != 0:
                        raise RuntimeError(
                            'Unrecognized charge {}'.format(charge))
                    if type != 'H':
                        g.add_node(nid, type=type)
                elif len(ls) == 6:
                    ls = line.rstrip().split()
                    nid0 = int(ls[0]) - 1
                    nid1 = int(ls[1]) - 1
                    valence = int(ls[2])
                    if nid0 in g.nodes() and nid1 in g.nodes():
                        g.add_edge(nid0, nid1, valence=valence)
                line_i += 1
    print(len(charges), charges)
    print(len(charge_gids), charge_gids)


def get_old_aids_id():
    files = glob(get_root_path() + '/data/AIDS_old/data/*.gxl')
    return [get_file_base_id(file) for file in files]


def gen_aids_small(name, additional=False):
    datadir = get_root_path() + '/data'
    dirin = datadir + '/AIDS40k_orig'
    sfn = get_save_path() + '/aids40k_orig'
    loaded = load_as_dict(sfn)
    if not loaded:
        graphs = {}
        nodes_graphs = defaultdict(list)
        lesseq30 = set()
        lesseq10 = set()
        disconnects = set()
        # Iterate through all 40k graphs.
        for file in glob(dirin + '/*.gexf'):
            gid = int(file.split('/')[-1].split('.')[0])
            g = nx.read_gexf(file)
            if not nx.is_connected(g):
                print('{} not connected'.format(gid))
                disconnects.add(gid)
                continue
            graphs[gid] = g
            nodes_graphs[g.number_of_nodes()].append(gid)
            if g.number_of_nodes() <= 30:
                lesseq30.add(gid)
            if g.number_of_nodes() <= 10:
                lesseq10.add(gid)
        save_as_dict(sfn, graphs, nodes_graphs, lesseq30, lesseq10, disconnects)
    else:
        graphs = loaded['graphs']
        nodes_graphs = loaded['nodes_graphs']
        lesseq30 = loaded['lesseq30']
        lesseq10 = loaded['lesseq10']
        disconnects = loaded['disconnects']
    print(len(disconnects), 'disconnected graphs out of', len(graphs))
    print(len(lesseq30), 'with <= 30 nodes')
    print(len(lesseq10), 'with <= 10 nodes')
    # exit(1)
    train_dir = '{}/{}/train'.format(datadir, name)
    if additional:
        train_data = load_data(name.lower(), train=True)
        test_dir_str = 'test2'
    else:
        exec('mkdir -p {}'.format(train_dir))
        test_dir_str = 'test'
    test_dir = '{}/{}/{}'.format(datadir, name, test_dir_str)
    exec('mkdir -p {}'.format(test_dir))
    if not additional:
        if name == 'AIDS10k':
            for num_node in range(5, 23):
                choose = random.Random(123).sample(nodes_graphs[num_node], 1)[0]
                print('choose {} with {} nodes'.format(choose, num_node))
                nx.write_gexf(
                    graphs[choose], test_dir + '/{}.gexf'.format(choose))
                lesseq30.remove(choose)
            for tid in random.Random(123).sample(lesseq30, 10000):
                nx.write_gexf(graphs[tid], train_dir + '/{}.gexf'.format(tid))
        elif name == 'AIDS700nef':
            lesseq10 = sample_from_lessthan10eq(
                train_dir, lesseq10, 560, graphs, 'train')
            sample_from_lessthan10eq(
                test_dir, lesseq10, 140, graphs, 'test')
    else:
        assert (name == 'AIDS10k')
        for num_node in range(5, 30):
            k = 4
            from_li = nodes_graphs[num_node]
            print('sampling {} from {} (size={})'.format(k, len(from_li),
                                                         num_node))
            choose = random.Random(123).sample_exclude(
                from_li, k, train_data.get_gids())
            print('choose {} with {} nodes'.format(choose, num_node))
            for c in choose:
                nx.write_gexf(graphs[c], test_dir + '/{}.gexf'.format(c))
    print('Done')


def sample_exclude(from_li, k, exclude):
    rtn = set()
    random.Random(123).shuffle(from_li)
    idx = 0
    for i in range(k):
        while True:
            c = from_li[idx]
            idx += 1
            if c not in rtn and c not in exclude:
                rtn.add(c)
                break
    return rtn


def sample_from_lessthan10eq(dir, lesseq10, num, graphs, s):
    for gid in sorted(random.Random(123).sample(lesseq10, num)):
        g = graphs[gid]
        print('{}: choose {} with {} nodes'.format(
            s, gid, g.number_of_nodes()))
        nx.write_gexf(g, dir + '/{}.gexf'.format(gid))
        lesseq10.remove(gid)
    return lesseq10


gen_aids_small(name='AIDS700nef', additional=False)
