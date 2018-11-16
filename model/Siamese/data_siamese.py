from data import Data
from config import FLAGS
from coarsening import coarsen, perm_data
from utils_siamese import get_coarsen_level
from utils import load_data, exec_turnoff_print
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder
import networkx as nx
import numpy as np


exec_turnoff_print()


class SiameseModelData(Data):
    def __init__(self):
        self.dataset = FLAGS.dataset
        self.valid_percentage = FLAGS.valid_percentage
        self.node_feat_name = FLAGS.node_feat_name
        self.node_feat_encoder = FLAGS.node_feat_encoder
        self.bsf_ordering = FLAGS.bfs_ordering
        self.coarsening = FLAGS.coarsening
        self.random_permute = FLAGS.random_permute
        super().__init__(self._get_name())
        print('{} train graphs; {} validation graphs; {} test graphs'.format(
            len(self.train_gs),
            len(self.val_gs),
            len(self.test_gs)))

    def init(self):
        orig_train_data = load_data(self.dataset, train=True)
        train_gs, val_gs = self._train_val_split(orig_train_data)
        test_gs = load_data(self.dataset, train=False).graphs
        self.node_feat_encoder = self._get_node_feature_encoder(
            orig_train_data.graphs + test_gs)
        self._check_graphs_num(test_gs, 'test')
        self.train_gs = [ModelGraph(g, self.node_feat_encoder) for g in train_gs]
        self.val_gs = [ModelGraph(g, self.node_feat_encoder) for g in val_gs]
        self.test_gs = [ModelGraph(g, self.node_feat_encoder) for g in test_gs]
        assert (len(train_gs) + len(val_gs) == len(orig_train_data.graphs))

    def input_dim(self):
        return self.node_feat_encoder.input_dim()

    def _get_name(self):
        li = []
        for k, v in sorted(self.__dict__.items(), key=lambda x: x[0]):
            li.append('{}'.format(v))
        return '_'.join(li)

    def _train_val_split(self, orig_train_data):
        if self.valid_percentage < 0 or self.valid_percentage > 1:
            raise RuntimeError('valid_percentage {} must be in [0, 1]'.format(
                self.valid_percentage))
        gs = orig_train_data.graphs
        sp = int(len(gs) * (1 - self.valid_percentage))
        train_graphs = gs[0:sp]
        valid_graphs = gs[sp:]
        self._check_graphs_num(train_graphs, 'train')
        self._check_graphs_num(valid_graphs, 'validation')
        return train_graphs, valid_graphs

    def _check_graphs_num(self, graphs, label):
        if len(graphs) <= 2:
            raise RuntimeError('Insufficient {} graphs {}'.format( \
                label, len(graphs)))

    def _get_node_feature_encoder(self, gs):
        if self.node_feat_encoder == 'onehot':
            return NodeFeatureOneHotEncoder(gs, self.node_feat_name)
        elif 'constant' in self.node_feat_encoder:
            return NodeFeatureConstantEncoder(gs, self.node_feat_name)
        else:
            raise RuntimeError('Unknown node_feat_encoder {}'.format(
                self.node_feat_encoder))


class ModelGraph(object):
    def __init__(self, nxgraph, node_feat_encoder):
        if FLAGS.bfs_ordering and FLAGS.coarsening:
            raise RuntimeError('Cannot use both bfs ordering and coarsening!')
        self.nxgraph = nxgraph
        self.dense_node_inputs = node_feat_encoder.encode(nxgraph)
        if FLAGS.random_permute:
            assert (not FLAGS.bfs_ordering and not FLAGS.coarsening)
            self.graph_size = len(nxgraph.nodes())
            self.permute_order = np.random.permutation(self.graph_size)
        if FLAGS.bfs_ordering:
            self.bfs_order = self._bfs_ordering(nxgraph)
            assert (len(self.bfs_order) == len(nxgraph.nodes()))
            self.dense_node_inputs = self.dense_node_inputs[self.bfs_order, :]
        self.sparse_node_inputs = self._preprocess_inputs(
            sp.csr_matrix(self.dense_node_inputs))
        # Only one laplacian.
        self.num_laplacians = 1
        self.adj = nx.to_numpy_matrix(nxgraph)
        if FLAGS.bfs_ordering:
            self.adj = self.adj[np.ix_(self.bfs_order, self.bfs_order)]
        if FLAGS.coarsening:
            self._coarsen()
        else:
            self.laplacians = [self._preprocess_adj(self.adj)]

    def get_nxgraph(self):
        return self.nxgraph

    def get_node_inputs(self):
        if FLAGS.coarsening:
            return self.sparse_permuted_padded_dense_node_inputs
        elif FLAGS.random_permute:
            dense_node_inputs = self.dense_node_inputs[self.permute_order, :]
            return self._preprocess_inputs(sp.csr_matrix(dense_node_inputs))
        else:
            return self.sparse_node_inputs

    def get_node_inputs_num_nonzero(self):
        return self.get_node_inputs()[1].shape

    def get_laplacians(self, gcn_id):
        if FLAGS.coarsening:
            return self.coarsened_laplacians[gcn_id]
        elif FLAGS.random_permute:
            adj = self.adj[np.ix_(self.permute_order, self.permute_order)]
            self._reset_permute()  # Assumption: 1st get_node_input then get_laplacians
            return [self._preprocess_adj(adj)]
        else:
            return self.laplacians

    def _coarsen(self):
        assert ('metis_' in FLAGS.coarsening)
        self.num_level = get_coarsen_level()
        assert (self.num_level >= 1)
        graphs, perm = coarsen(sp.csr_matrix(self.adj), levels=self.num_level,
                               self_connections=False)
        self.permuted_padded_dense_node_inputs = perm_data(
            self.dense_node_inputs.T, perm).T
        self.sparse_permuted_padded_dense_node_inputs = self._preprocess_inputs(
            sp.csr_matrix(self.permuted_padded_dense_node_inputs))
        self.coarsened_laplacians = []
        for g in graphs:
            self.coarsened_laplacians.append([self._preprocess_adj(g.todense())])
        assert (len(self.coarsened_laplacians) == self.num_laplacians * self.num_level + 1)

    def _preprocess_inputs(self, inputs):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(inputs.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        inputs = r_mat_inv.dot(inputs)
        return self._sparse_to_tuple(inputs)

    def _preprocess_adj(self, adj):
        """Preprocessing of adjacency matrix and conversion to tuple representation."""
        adj_normalized = self._normalize_adj(adj + sp.eye(adj.shape[0]))
        return self._sparse_to_tuple(adj_normalized)

    def _normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def _sparse_to_tuple(self, sparse_mx):
        """Convert sparse matrix to tuple representation."""

        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx

    def _bfs_seq(self, graph, start_id):
        dictionary = dict(nx.bfs_successors(graph, start_id))
        start = [start_id]
        output = [start_id]
        while len(start) > 0:
            next = []
            while len(start) > 0:
                current = start.pop(0)
                neighbor = dictionary.get(current)
                if neighbor is not None:
                    neighbor.sort()
                    next += neighbor
            output += next
            start = next
        return output

    def _bfs_ordering(self, graph):
        degree_dict = {}
        for i in range(len(graph.nodes())):
            degree_dict[graph.nodes()[i]] = graph.degree(graph.nodes()[i])
        degree_list = sorted(degree_dict.items(), key=lambda item: item[1], reverse=True)
        bfs_seq = [int(idx) for idx in self._bfs_seq(graph, degree_list[0][0])]
        origin_seq = [int(n) for n in graph.nodes()]
        bfs_order = []
        for e in bfs_seq:
            bfs_order.append(origin_seq.index(e))
        return bfs_order

    def _reset_permute(self):
        self.permute_order = np.random.permutation(self.graph_size)


class NodeFeatureEncoder(object):
    def encode(self, g):
        raise NotImplementedError()

    def input_dim(self):
        raise NotImplementedError()


class NodeFeatureOneHotEncoder(NodeFeatureEncoder):
    def __init__(self, gs, node_feat_name):
        self.node_feat_name = node_feat_name
        inputs_set = set()
        for g in gs:
            inputs_set = inputs_set | set(self._node_feat_dic(g).values())
        self.feat_idx_dic = {feat: idx for idx, feat in enumerate(inputs_set)}
        self.oe = OneHotEncoder().fit(
            np.array(list(self.feat_idx_dic.values())).reshape(-1, 1))

    def encode(self, g):
        node_feat_dic = self._node_feat_dic(g)
        temp = [self.feat_idx_dic[node_feat_dic[n]] for n in g.nodes()]
        return self.oe.transform(np.array(temp).reshape(-1, 1)).toarray()

    def input_dim(self):
        return self.oe.transform([[0]]).shape[1]

    def _node_feat_dic(self, g):
        return nx.get_node_attributes(g, self.node_feat_name)


class NodeFeatureConstantEncoder(NodeFeatureEncoder):
    def __init__(self, _, node_feat_name):
        self.input_dim_ = int(FLAGS.node_feat_encoder.split('_')[1])
        self.const = float(2.0)
        assert (node_feat_name is None)

    def encode(self, g):
        rtn = np.full((g.number_of_nodes(), self.input_dim_), self.const)
        return rtn

    def input_dim(self):
        return self.input_dim_
