from utils import get_result_path, get_save_path, get_model_path, \
    get_file_base_id, load_data, load_pkl, save_pkl
from distance import normalized_ged
from classification import get_classification_labels_from_dist_mat
from similarity import create_sim_kernel
from glob import glob
import numpy as np
import json
from os.path import isfile


class Result(object):
    """
    The result object loads and stores the ranking result of a model
        for evaluation.
        Terminology:
            rtn: return value of a function.
            m: # of queries.
            n: # of database graphs.
    """

    def model(self):
        """
        :return: The model name.
        """
        return self.model_

    def m_n(self):
        return self.dist_sim_mat(norm=False).shape

    def dist_or_sim(self):
        raise NotImplementedError()

    def dist_sim_mat(self, norm):
        """
        Each result object stores either a distance matrix
            or a similarity matrix. It cannot store both.
        :param norm:
        :return: either the distance matrix or the similairty matrix.
        """
        raise NotImplementedError()

    def dist_sim(self, qid, gid, norm):
        """
        :param qid: query id (0-indexed).
        :param gid: database graph id (0-indexed) (NOT g.graph['gid']).
        :param norm:
        :return: (metric, dist or sim between qid and gid)
        """
        raise NotImplementedError()

    def sim_mat(self, sim_kernel, yeta, scale, norm):
        raise NotImplementedError()

    def top_k_ids(self, qid, k, norm, inclusive, rm):
        """
        :param qid: query id (0-indexed).
        :param k: 
        :param norm: 
        :param inclusive: whether to be tie inclusive or not.
            For example, the ranking may look like this:
            7 (sim_score=0.99), 5 (sim_score=0.99), 10 (sim_score=0.98), ...
            If tie inclusive, the top 1 results are [7, 9].
            Therefore, the number of returned results may be larger than k.
            In summary,
                len(rtn) == k if not tie inclusive;
                len(rtn) >= k if tie inclusive.
        :return: for a query, the ids of the top k database graph
        ranked by this model.
        """
        sort_id_mat = self.sort_id_mat(norm)
        _, n = sort_id_mat.shape
        if k < 0 or k >= n:
            raise RuntimeError('Invalid k {}'.format(k))
        if not inclusive:
            return sort_id_mat[qid][:k]
        # Tie inclusive.
        dist_sim_mat = self.dist_sim_mat(norm)
        while k < n:
            cid = sort_id_mat[qid][k - 1]
            nid = sort_id_mat[qid][k]
            if abs(dist_sim_mat[qid][cid] - dist_sim_mat[qid][nid]) <= rm:
                k += 1
            else:
                break
        return sort_id_mat[qid][:k]

    def ranking(self, qid, gid, norm, one_based=True):
        """
        :param qid: query id (0-indexed).
        :param gid: database graph id (0-indexed) (NOT g.graph['gid']).
        :param norm:
        :param one_based: whether to return the 1-based or 0-based rank.
            True by default.
        :return: for a query, the rank of a database graph by this model.
        """
        # Assume self is ground truth.
        sort_id_mat = self.sort_id_mat(norm)
        finds = np.where(sort_id_mat[qid] == gid)
        assert (len(finds) == 1 and len(finds[0]) == 1)
        fid = finds[0][0]
        # Tie inclusive (always when find ranking).
        dist_sim_mat = self.dist_sim_mat(norm)
        while fid > 0:
            cid = sort_id_mat[qid][fid]
            pid = sort_id_mat[qid][fid - 1]
            if dist_sim_mat[qid][pid] == dist_sim_mat[qid][cid]:
                fid -= 1
            else:
                break
        if one_based:
            fid += 1
        return fid

    def classification_mat(self, thresh_pos, thresh_neg,
                           thresh_pos_sim, thresh_neg_sim, norm):
        raise NotImplementedError()

    def time(self, qid, gid):
        raise NotImplementedError()

    def time_mat(self):
        raise NotImplementedError()

    def mat(self, metric, norm):
        raise NotImplementedError()

    def sort_id_mat(self, norm):
        """
        :param norm:
        :return: a m by n matrix representing the ranking result.
            rtn[i][j]: For query i, the id of the j-th most similar
                       graph ranked by this model.
        """
        raise NotImplementedError()

    def ranking_mat(self, norm, one_based=True):
        """
        :param norm:
        :param one_based:
        :return: a m by n matrix representing the ranking result.
                 Note it is different from sort_id_mat.
            rtn[i][j]: For query i, the ranking of the graph j.
        """
        m, n = self.m_n()
        rtn = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                rtn[i][j] = self.ranking(i, j, norm, one_based=one_based)
        return rtn


class DistanceModelResult(Result):
    def __init__(self, dataset, model, dist_mat, row_graphs, col_graphs):
        self.dataset = dataset
        self.model_ = model
        if dist_mat is not None:
            self.dist_mat_ = dist_mat
            self.time_mat_ = None
        else:
            self.dist_mat_ = self._load_result_mat(
                self.dist_metric(), self.model_, len(row_graphs), len(col_graphs))
            self.time_mat_ = self._load_result_mat(
                'time', self.model_, len(row_graphs), len(col_graphs))
        self.dist_norm_mat_ = np.copy(self.dist_mat_)
        m, n = self.dist_mat_.shape
        assert (m == len(row_graphs))
        assert (n == len(col_graphs))
        for i in range(m):
            for j in range(n):
                self.dist_norm_mat_[i][j] = normalized_ged(
                    self.dist_mat_[i][j], row_graphs[i], col_graphs[j])
        self.sort_id_mat_ = np.argsort(self.dist_mat_, kind='mergesort')
        self.dist_norm_sort_id_mat_ = np.argsort(self.dist_norm_mat_, kind='mergesort')

    def dist_metric(self):
        raise NotImplementedError()

    def dist_mat(self, norm):
        return self._select_dist_mat(norm)

    def dist_sim_mat(self, norm):
        return self.dist_mat(norm)

    def dist_or_sim(self):
        return 'dist'

    def dist_sim(self, qid, gid, norm):
        return self.dist_metric(), self._select_dist_mat(norm)[qid][gid]

    def sim_mat(self, sim_kernel, yeta, scale, norm):
        rtn = create_sim_kernel(sim_kernel, yeta, scale). \
            dist_to_sim_np(self.dist_mat(norm))
        return rtn

    def classification_mat(self, thresh_pos, thresh_neg,
                           thresh_pos_sim, thresh_neg_sim, norm):
        vname = 'classif_mat_{}_{}_{}'.format(thresh_pos, thresh_neg, norm)
        if hasattr(self, vname):
            return getattr(self, vname)
        dist_mat = self.dist_sim_mat(norm)
        label_mat, num_poses, num_negs, _, _ = \
            get_classification_labels_from_dist_mat(
                dist_mat, thresh_pos, thresh_neg)
        rtn = (label_mat, num_poses, num_negs)
        setattr(self, vname, rtn)
        return rtn

    def time(self, qid, gid):
        return self.time_mat_[qid][gid]

    def time_mat(self):
        return self.time_mat_

    def mat(self, metric, norm):
        if metric == self.dist_metric():
            return self._select_dist_mat(norm)
        elif metric == 'time':
            return self.time_mat_
        else:
            raise RuntimeError('Unknown metric {} for model {}'.format(
                metric, self.model_))

    def sort_id_mat(self, norm):
        return self._select_sort_id_mat(norm)

    def _load_result_mat(self, metric, model, m, n):
        file_p = get_result_path() + '/{}/{}/{}_{}_mat_{}_{}_*.npy'.format(
            self.dataset, metric, self.dist_metric(), metric, self.dataset,
            model)
        li = glob(file_p)
        if not li:
            if 'astar' in model:
                if self.dataset != 'imdbmulti':
                    raise RuntimeError('Not imdbmulti and no astar results!')
                return self._load_merged_astar_from_other_three(metric, m, n)
            else:
                raise RuntimeError('No results found {}'.format(file_p))
        file = self._choose_result_file(li, m, n)
        return np.load(file)

    def _load_merged_astar_from_other_three(self, metric, m, n):
        self.model_ = 'merged_astar'
        merge_models = ['beam80', 'hungarian', 'vj']
        dms = [self._load_result_mat(self.dist_metric(), model, m, n)
               for model in merge_models]
        tms = [self._load_result_mat('time', model, m, n)
               for model in merge_models]
        for i in range(len(merge_models)):
            assert (dms[i].shape == (m, n))
            assert (tms[i].shape == (m, n))
        rtn = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                d_li = [dm[i][j] for dm in dms]
                d_min = np.min(d_li)
                if metric == self.dist_metric():
                    rtn[i][j] = d_min
                else:
                    assert (metric == 'time')
                    d_argmin = np.argmin(d_li)
                    rtn[i][j] = tms[d_argmin][i][j]
        return rtn

    def _choose_result_file(self, files, m, n):
        cands = []
        for file in files:
            temp = np.load(file)
            if temp.shape == (m, n):
                cands.append(file)
                if 'qilin' in file:
                    print(file)
                    return file
        if cands:
            return cands[0]
        raise RuntimeError('No result files in {}'.format(files))  # TODO: smart choice and cross-checking

    def _select_dist_mat(self, norm):
        return self.dist_norm_mat_ if norm else self.dist_mat_

    def _select_sort_id_mat(self, norm):
        return self.dist_norm_sort_id_mat_ if norm else self.sort_id_mat_


class PairwiseGEDModelResult(DistanceModelResult):
    def dist_metric(self):
        return 'ged'


class PairwiseMCSModelResult(DistanceModelResult):
    def dist_metric(self):
        return 'mcs'


class SimilarityBasedModelResult(Result):
    def sim_mat(self, sim_kernel=None, yeta=None, scale=None, norm=None):
        return self.sim_mat_

    def dist_sim_mat(self, norm=False):
        return self.sim_mat()

    def dist_or_sim(self):
        return 'sim'

    def dist_sim(self, qid, gid, norm=False):
        return 'sim', self.sim_mat_[qid][gid]

    def classification_mat(self, thresh_pos, thresh_neg,
                           thresh_pos_sim, thresh_neg_sim, norm):
        m, n = self.m_n()
        sim_mat = self.sim_mat()
        label_mat = np.zeros((m, n))
        num_poses = 0
        num_negs = 0
        for i in range(m):
            num_pos = 0
            num_neg = 0
            for j in range(n):
                s = sim_mat[i][j]
                if s >= thresh_pos_sim:
                    label_mat[i][j] = 1
                    num_pos += 1
                elif s < thresh_neg_sim:
                    label_mat[i][j] = -1
                    num_neg += 1
            num_poses += num_pos
            num_negs += num_neg
        return label_mat, num_poses, num_negs

    def sort_id_mat(self, norm=False):
        return self.sort_id_mat_

    def _gen_sort_id_mat(self):
        # Reverse the sorting since similarity-based.
        # More similar items come first.
        return np.argsort(self.sim_mat_, kind='mergesort')[:, ::-1]


class Graph2VecResult(SimilarityBasedModelResult):
    def __init__(self, dataset, model, sim):
        self.dim = 1024
        self.model_ = model
        self.dataset = dataset
        self.sim = sim
        self.sim_mat_ = self._load_sim_mat()
        self.sort_id_mat_ = self._gen_sort_id_mat()

    def time(self, qid, gid):
        return None

    def mat(self, metric, *unused):
        if metric == 'sim':
            return self.sim_mat_
        else:
            raise RuntimeError('Unknown metric {} for model {}'.format(
                metric, self.model_))

    def _load_sim_mat(self):
        fn = get_result_path() + '/{}/sim/{}_graph2vec_dim_{}_sim_{}.npy'.format(
            self.dataset, self.dataset, self.dim, self.sim)
        if isfile(fn):
            with open(fn, 'rb') as handle:
                sim_mat = load_pkl(handle)
                print('Loaded sim mat from {}'.format(fn))
                return sim_mat
        train_emb = self._load_emb(True)
        test_emb = self._load_emb(False)
        if self.sim == 'dot':
            sim_mat = test_emb.dot(train_emb.T)
        else:
            raise RuntimeError('Unknown sim {}'.format(self.sim))
        with open(fn, 'wb') as handle:
            save_pkl(sim_mat, handle)
            print('Saved sim mat {} to {}'.format(sim_mat.shape, fn))
        return sim_mat

    def _load_emb(self, train):
        fn = get_result_path() + '/{}/emb/{}_graph2vec_{}_emb_dim_{}.npy'.format(
            self.dataset, self.dataset, 'train' if train else 'test', self.dim)
        if isfile(fn):
            emb = np.load(fn)
            print('Loaded emb {} from {}'.format(emb.shape, fn))
            return emb
        data = load_data(self.dataset, train=train)
        id_map = self._gid_to_matrixid(data)
        emb = np.zeros((len(data.graphs), self.dim))
        cnt = 0
        d = self._load_json_emb()
        for f in d:
            gid = get_file_base_id(f)
            if gid in id_map:
                emb[id_map[gid]] = d[f]
                cnt += 1
        if cnt != len(id_map):
            raise RuntimeError('Mismatch: {} != {}').format(
                cnt, len(id_map))
        np.save(fn, emb)
        print('Saved emb {} to {}'.format(emb.shape, fn))
        return emb

    def _load_json_emb(self):
        fn = get_save_path() + '/{}_graph2vec_json_dict.pkl'.format(
            self.dataset)
        if isfile(fn):
            with open(fn, 'rb') as handle:
                d = load_pkl(handle)
                print('Loaded json dict from {}'.format(fn))
                return d
        dfn = get_model_path() + '/graph2vec_tf/embeddings/{}_train_test_dims_{}_epochs_1000_lr_0.3_embeddings.txt'.format(
            self.dataset, self.dim)
        with open(dfn) as json_data:
            d = json.load(json_data)
        with open(fn, 'wb') as handle:
            save_pkl(d, handle)
            print('Loaded json dict from {}\nSaved to {}'.format(dfn, fn))
        return d

    def _gid_to_matrixid(self, data):
        rtn = {}
        for i, g in enumerate(data.graphs):
            rtn[g.graph['gid']] = i
        return rtn


class SiameseModelResult(SimilarityBasedModelResult):
    def __init__(self, dataset, model,
                 sim_mat=None, time_mat=None, model_info=None):
        self.model_ = model
        self.dataset = dataset
        if sim_mat is not None:
            self.sim_mat_ = sim_mat
        else:
            self.sim_mat_ = self._load_sim_mat(model_info)
        if time_mat is not None:
            self.time_mat_ = time_mat
        else:
            self.time_mat_ = self._load_time_mat(model_info)
        self.sort_id_mat_ = self._gen_sort_id_mat()

    def time(self, qid, gid):
        return self.time_mat_[qid][gid]

    def time_mat(self):
        return self.time_mat_

    def mat(self, metric, *unused):
        if metric == 'sim':
            return self.sim_mat_
        elif metric == 'time':
            return self.time_mat_
        else:
            raise RuntimeError('Unknown metric {} for model {}'.format(
                metric, self.model_))

    def _load_sim_mat(self, model_info):
        raise NotImplementedError()

    def _load_time_mat(self, model_info):
        raise NotImplementedError()


class TransductiveResult(SimilarityBasedModelResult):
    def __init__(self, dataset, model,
                 sim_mat=None, model_info=None):
        self.model_ = model
        self.dataset = dataset
        if sim_mat is not None:
            self.sim_mat_ = sim_mat
        else:
            self.sim_mat_ = self._load_sim_mat(model_info)
        self.sort_id_mat_ = self._gen_sort_id_mat()

    def mat(self, metric, *unused):
        if metric == 'sim':
            return self.sim_mat_
        else:
            raise RuntimeError('Unknown metric {} for model {}'.format(
                metric, self.model_))

    def _load_sim_mat(self, model_info):
        raise NotImplementedError()

    def _load_time_mat(self, model_info):
        raise NotImplementedError()


def load_results_as_dict(dataset, models, sim='dot',
                         sim_mat=None, dist_mat=None,
                         row_graphs=None, col_graphs=None,
                         time_mat=None, model_info=None):
    rtn = {}
    for model in models:
        rtn[model] = load_result(
            dataset, model, sim,
            sim_mat=sim_mat, dist_mat=dist_mat,
            row_graphs=row_graphs, col_graphs=col_graphs,
            time_mat=time_mat, model_info=model_info)
    return rtn


def load_result(dataset, model, sim=None, sim_mat=None, dist_mat=None,
                row_graphs=None, col_graphs=None,
                time_mat=None, model_info=None):
    if 'beam' in model or model in ['astar', 'hungarian', 'vj']:
        return PairwiseGEDModelResult(
            dataset, model, dist_mat, row_graphs, col_graphs)
    elif model == 'graph2vec':
        return Graph2VecResult(dataset, model, sim)
    elif 'siamese' in model:
        return SiameseModelResult(dataset, model, sim_mat, time_mat, model_info)
    elif 'transductive' in model:
        return TransductiveResult(dataset, model, sim_mat, model_info)
    else:
        raise RuntimeError('Unknown model {}'.format(model))


if __name__ == '__main__':
    Graph2VecResult('aids10k', 'graph2vec', sim='dot')
