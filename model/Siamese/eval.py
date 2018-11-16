from config import FLAGS
from utils_siamese import get_flags
from utils import load_data
from exp import BASELINE_MODELS, plot_preck, plot_single_number_metric, \
    visualize_embeddings_binary, visualize_embeddings_gradual,\
    draw_attention, draw_emb_hist_heat, comb_gt_rk
from metrics import auc
from dist_calculator import get_gs_dist_mat
from results import load_results_as_dict, load_result
import numpy as np
from collections import OrderedDict


class Eval(object):
    def __init__(self, data, dist_calculator):
        test_gs1 = load_data(FLAGS.dataset, train=False).graphs
        test_gs2 = load_data(FLAGS.dataset, train=True).graphs
        self.baseline_models = BASELINE_MODELS
        if FLAGS.dataset == 'imdbmulti':
            self.baseline_models = ['hungarian', 'vj', 'beam80', 'beam1', 'beam2']
        self.baseline_results_dict = load_results_as_dict(
            FLAGS.dataset, self.baseline_models,
            row_graphs=test_gs1, col_graphs=test_gs2)
        val_gs1, val_gs2 = self.get_val_gs_as_tuple(data)
        self.true_val_result = load_result(
            FLAGS.dataset, FLAGS.dist_algo,
            dist_mat=self._get_true_dist_mat_for_val(data, dist_calculator),
            row_graphs=self._to_nxgraph_list(val_gs1),
            col_graphs=self._to_nxgraph_list(val_gs2))
        self.true_test_result = load_result(
            FLAGS.dataset, FLAGS.dist_algo
            , row_graphs=test_gs1, col_graphs=test_gs2)
        self.norms = [FLAGS.dist_norm]

    def get_val_gs_as_tuple(self, data):
        return data.val_gs, data.train_gs

    def get_test_gs_as_tuple(self, data):
        return data.test_gs, data.train_gs + data.val_gs

    def _get_true_dist_mat_for_val(self, data, dist_calculator):
        gs1, gs2 = self.get_val_gs_as_tuple(data)
        gs1 = self._to_nxgraph_list(gs1)
        gs2 = self._to_nxgraph_list(gs2)
        return get_gs_dist_mat(gs1, gs2, dist_calculator, 'val', 'train',
                               FLAGS.dataset, FLAGS.dist_metric,
                               FLAGS.dist_algo, norm=False)

    def _to_nxgraph_list(self, gs):
        return [g.nxgraph for g in gs]

    def get_true_sim(self, i, j, val_or_test, model):
        if val_or_test == 'val':
            r = self.true_val_result
        else:
            assert (val_or_test == 'test')
            r = self.true_test_result
        return model.get_true_sim(i, j, r)

    def eval_for_val(self, sim_mat, loss_list, time_list, metrics):
        models = [FLAGS.model]
        pred_r = load_result(
            FLAGS.dataset, FLAGS.model, sim_mat=sim_mat, time_mat=time_list)
        rs = {FLAGS.model: pred_r, FLAGS.dist_algo: self.true_val_result}
        results = self._eval(models, rs, self.true_val_result,
                             loss_list, metrics, False)
        rtn = OrderedDict()
        li = []
        for metric, num in results.items():
            if not 'loss' in metric:
                num = num[FLAGS.model]
                results[metric] = num
            metric = 'val_' + self._remove_norm_from_str(metric)
            rtn[metric] = num
            s = '{}={:.5f}'.format(metric, num)
            li.append(s)
        return rtn, ' '.join(li)

    def eval_for_test(self, sim_mat, loss_list, time_list,
                      node_embs_list, graph_embs_mat, attentions,
                      metrics, saver):
        models = [FLAGS.dist_algo] + self.baseline_models + [FLAGS.model]
        rs = {FLAGS.model: load_result(
            FLAGS.dataset, FLAGS.model, sim_mat=sim_mat, time_mat=time_list),
            FLAGS.dist_algo: self.true_test_result}
        if FLAGS.plot_results:
            rs.update(self.baseline_results_dict)
        eps_dir = saver.get_log_dir()
        return self._eval(models, rs, self.true_test_result,
                          loss_list, metrics, FLAGS.plot_results,
                          node_embs_list, graph_embs_mat, attentions,
                          eps_dir)

    def _eval(self, models, rs, true_r, loss_list, metrics, plot,
              node_embs_list=None, graph_embs_mat=None, attentions=None,
              eps_dir=None):
        rtn = OrderedDict()
        for metric in metrics:
            if metric == 'mrr' or metric == 'mse' or metric == 'time' or \
                    'acc' in metric or metric == 'kendalls_tau' or \
                    metric == 'spearmans_rho':
                d = plot_single_number_metric(
                    FLAGS.dataset, models, rs, true_r, metric,
                    self.norms,
                    sim_kernel=get_flags('sim_kernel'),
                    yeta=get_flags('yeta'),
                    scale=get_flags('scale'),
                    thresh_poss=[get_flags('thresh_val_test_pos')],
                    thresh_negs=[get_flags('thresh_val_test_neg')],
                    thresh_poss_sim=[0.5],
                    thresh_negs_sim=[0.5],
                    plot_results=plot, eps_dir=eps_dir)
                rtn.update(d)
            elif metric == 'draw_gt_rk':
                comb_gt_rk(FLAGS.dataset, FLAGS.dist_algo,
                           rs[FLAGS.model], eps_dir + '/gt_rk')
            elif metric == 'groundtruth':
                pass
            elif metric == 'draw_heat_hist':
                if node_embs_list is not None:
                    draw_emb_hist_heat(
                        FLAGS.dataset, node_embs_list, FLAGS.dist_norm,
                        max_nodes=FLAGS.max_nodes,
                        apply_sigmoid=True,
                        eps_dir=eps_dir + '/mne')
            elif metric == 'emb_vis_gradual':
                if graph_embs_mat is not None:
                    visualize_embeddings_gradual(
                        FLAGS.dataset,
                        graph_embs_mat,
                        eps_dir=eps_dir + '/emb_vis_gradual')
            elif metric == 'ranking':
                pass
                # ranking(
                #     FLAGS.dataset, FLAGS.dist_algo, rs[FLAGS.model],
                #     eps_dir=eps_dir + '/ranking'
                # )
            elif metric == 'attention':
                if attentions is not None:
                    draw_attention(
                        FLAGS.dataset, FLAGS.dist_algo, attentions,
                        eps_dir=eps_dir + '/attention')
            elif metric == 'auc':
                auc_score = auc(
                    true_r, rs[FLAGS.model],
                    thresh_pos=
                    get_flags('thresh_val_test_pos'),
                    thresh_neg=
                    get_flags('thresh_val_test_neg'),
                    norm=FLAGS.dist_norm)
                print('auc', auc_score)
                rtn.update({'auc': auc_score})
            elif 'prec@k' in metric:
                d = plot_preck(
                    FLAGS.dataset, models, rs, true_r, metric,
                    self.norms, plot, eps_dir=eps_dir)
                rtn.update(d)
            elif metric == 'loss':
                rtn.update({metric: np.mean(loss_list)})
            elif metric == 'emb_vis_binary':
                if graph_embs_mat is not None:
                    visualize_embeddings_binary(
                        FLAGS.dataset, graph_embs_mat,
                        self.true_test_result,
                        thresh_pos=
                        get_flags('thresh_val_test_pos'),
                        thresh_neg=
                        get_flags('thresh_val_test_neg'),
                        thresh_pos_sim=0.5,
                        thresh_neg_sim=0.5,
                        norm=FLAGS.dist_norm,
                        eps_dir=eps_dir + '/emb_vis_binary')
            else:
                raise RuntimeError('Unknown metric {}'.format(metric))
        return rtn

    def _remove_norm_from_str(self, s):
        return s.replace('_norm', '').replace('_nonorm', '')
