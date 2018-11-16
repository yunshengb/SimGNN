from utils import get_train_str, get_data_path, get_save_path, sorted_nicely, \
    save, load
import networkx as nx
import random
from random import randint
from glob import glob
from os.path import basename


class Data(object):
    def __init__(self, name_str):
        name = self.__class__.__name__ + '_' + name_str + self.name_suffix()
        self.name = name
        sfn = self.save_filename(self.name)
        temp = load(sfn)
        if temp:
            self.__dict__ = temp
            print('{} loaded from {}{}'.format(
                name, sfn,
                ' with {} graphs'.format(
                    len(self.graphs)) if
                hasattr(self, 'graphs') else ''))
        else:
            self.init()
            save(sfn, self.__dict__)
            print('{} saved to {}'.format(name, sfn))

    def init(self):
        raise NotImplementedError()

    def name_suffix(self):
        return ''

    def save_filename(self, name):
        return '{}/{}'.format(get_save_path(), name)

    def get_gids(self):
        return [g.graph['gid'] for g in self.graphs]


class SynData(Data):
    train_num_graphs = 20
    test_num_graphs = 10

    def __init__(self, train):
        if train:
            self.num_graphs = SynData.train_num_graphs
        else:
            self.num_graphs = SynData.test_num_graphs
        super().__init__(get_train_str(train))

    def init(self):
        self.graphs = []
        for i in range(self.num_graphs):
            n = randint(5, 20)
            m = randint(n - 1, n * (n - 1) / 2)
            g = nx.gnm_random_graph(n, m)
            g.graph['gid'] = i
            self.graphs.append(g)
        print('Randomly generated %s graphs' % self.num_graphs)

    def name_suffix(self):
        return '_{}_{}'.format(SynData.train_num_graphs,
                               SynData.test_num_graphs)


class AIDSData(Data):
    def __init__(self, train):
        self.train = train
        super().__init__(get_train_str(train))

    def init(self):
        self.graphs = []
        datadir = '{}/{}/{}'.format(
            get_data_path(), self.get_folder_name(), get_train_str(self.train))
        self.graphs = iterate_get_graphs(datadir)
        print('Loaded {} graphs from {}'.format(len(self.graphs), datadir))
        if 'nef' in self.get_folder_name():
            print('Removing edge features')
            for g in self.graphs:
                self._remove_valence(g)

    def get_folder_name(self):
        raise NotImplementedError()

    def _remove_valence(self, g):
        for n1, n2, d in g.edges(data=True):
            d.pop('valence', None)


class AIDS10kData(AIDSData):
    def get_folder_name(self):
        return 'AIDS10k'


class AIDS10knefData(AIDS10kData):
    def init(self):
        self.graphs = AIDS10kData(self.train).graphs
        for g in self.graphs:
            self._remove_valence(g)
        print('Processed {} graphs: valence removed'.format(len(self.graphs)))


class AIDS700nefData(AIDSData):
    def get_folder_name(self):
        return 'AIDS700nef'

    def _remove_valence(self, g):
        for n1, n2, d in g.edges(data=True):
            d.pop('valence', None)


class AIDS80nefData(AIDS700nefData):
    def init(self):
        self.graphs = AIDS700nefData(self.train).graphs
        random.Random(123).shuffle(self.graphs)
        if self.train:
            self.graphs = self.graphs[0:70]
        else:
            self.graphs = self.graphs[0:10]
        print('Loaded {} graphs: valence removed'.format(len(self.graphs)))


class LinuxData(Data):
    def __init__(self, train):
        self.train = train
        super().__init__(get_train_str(train))

    def init(self):
        self.graphs = []
        datadir = '{}/linux/{}'.format(
            get_data_path(), get_train_str(self.train))
        self.graphs = iterate_get_graphs(datadir)
        print('Loaded {} graphs from {}'.format(len(self.graphs), datadir))


class IMDB1kData(Data):
    def __init__(self, train):
        self.train = train
        super().__init__(get_train_str(train))

    def init(self):
        self.graphs = []
        datadir = '{}/IMDB1k{}/{}'.format(
            get_data_path(), self._identity(), get_train_str(self.train))
        self.graphs = iterate_get_graphs(datadir)
        print('Loaded {} graphs from {}'.format(len(self.graphs), datadir))

    def _identity(self):
        raise NotImplementedError()


class IMDB1kCoarseData(IMDB1kData):
    def _identity(self):
        return 'Coarse'


class IMDB1kFineData(IMDB1kData):
    def _identity(self):
        return 'Fine'


class IMDBMultiData(Data):
    def __init__(self, train):
        self.train = train
        super().__init__(get_train_str(train))

    def init(self):
        self.graphs = []
        datadir = '{}/IMDBMulti/{}'.format(
            get_data_path(), get_train_str(self.train))
        self.graphs = iterate_get_graphs(datadir)
        print('Loaded {} graphs from {}'.format(len(self.graphs), datadir))


class IMDBMulti800Data(Data):
    def __init__(self, train):
        self.train = train
        super().__init__(get_train_str(train))

    def init(self):
        self.graphs = []
        datadir = '{}/IMDBMulti800/{}'.format(
            get_data_path(), get_train_str(self.train))
        self.graphs = iterate_get_graphs(datadir)
        print('Loaded {} graphs from {}'.format(len(self.graphs), datadir))


def iterate_get_graphs(dir):
    graphs = []
    for file in sorted_nicely(glob(dir + '/*.gexf')):
        gid = int(basename(file).split('.')[0])
        g = nx.read_gexf(file)
        g.graph['gid'] = gid
        graphs.append(g)
        if not nx.is_connected(g):
            raise RuntimeError('{} not connected'.format(gid))
    return graphs


if __name__ == '__main__':
    from utils import load_data
    nn = []
    data = load_data('imdbmulti', False)
    for g in data.graphs:
        nn.append(g.number_of_nodes())
        print(g.graph['gid'], g.number_of_nodes())
    print(max(nn))
