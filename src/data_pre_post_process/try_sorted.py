import sys
import networkx as nx
sys.path.append('../')
from utils import load_data


def bfs_seq(G, start_id, label_dict, weight):
   '''
   get a bfs node sequence
   :param G:
   :param start_id:
   :return:
   '''
   dictionary = dict(nx.bfs_successors(G, start_id))
   start = [start_id]
   output = [start_id]
   while len(start) > 0:
       next = []
       while len(start) > 0:
           current = start.pop(0)
           neighbor = dictionary.get(current)
           if neighbor is not None:
               #### a wrong example, should not permute here!
               # shuffle(neighbor)
               neighbor = sorted(neighbor, key=lambda item: (weight[label_dict[item]], -int(item)), reverse=True)
               next = next + neighbor
       output = output + next
       start = next
   return output


if __name__ == '__main__':
    aids700nef_weight = {'S': 423, 'Se': 2, 'O': 1051, 'As': 5,
                         'C': 3428, 'N': 1098, 'Si': 12, 'Br': 29, 'F': 14, 'Hg': 1,
                         'Ru': 2, 'B': 11, 'Pb': 1, 'Cu': 1, 'Sb': 1,
                         'Bi': 1, 'Co': 2, 'Ho': 1, 'Pd': 2, 'P': 21, 'Sn': 1,
                         'Ga': 1, 'Te': 1, 'I': 10, 'Ni': 1,
                         'Tb': 1, 'Li': 2, 'Pt': 6, 'Cl': 101}
    imdb_weight = {'actor': 2848, 'movie': 939, 'writer': 1527, 'producer': 728,
                   'editor': 146, 'cinematographer': 151, 'self': 9, 'actress': 1302,
                   'tvSeries': 60, 'composer': 234, 'director': 1011,
                   'archive_footage': 1, 'tvMiniSeries': 1, 'production_designer': 39}
    G = nx.read_gexf('../../data/AIDS700nef/train/5327.gexf')
    print(G.nodes())
    print(G.nodes(data=True))

    relabel_dict = {}
    #relabel
    for i in range(len(G.nodes(data=True))):
        relabel_dict[G.nodes(data=True)[i][1]['label']] = str(i)
        G.nodes(data=True)[i][1]['label'] = str(i)
    print(relabel_dict)
    G = nx.relabel_nodes(G, relabel_dict)
    print(G.nodes(data=True))

    degree_dict = {}
    label_dict = {}
    for i in range(len(G.nodes())):
        degree_dict[G.nodes()[i]] = G.degree(G.nodes()[i])
    for i in range(len(G.nodes(data=True))):
        label_dict[G.nodes(data=True)[i][0]] = G.nodes(data=True)[i][1]['type']
    # degree
    degree_list = sorted(degree_dict.items(), key=lambda item: item[1], reverse=True)
    # label
    label_list = sorted(label_dict.items(), key=lambda item: (aids700nef_weight[item[1]], -int(item[0])),
                        reverse=True)
    output = bfs_seq(G, label_list[0][0], label_dict, aids700nef_weight)
    print(output)
