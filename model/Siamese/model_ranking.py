from config import FLAGS
from model import Model
import numpy as np
from samplers import SelfShuffleList
from utils_siamese import get_coarsen_level
from dist_calculator import get_gs_dist_mat
import tensorflow as tf


class SiameseRankingModel(Model):
    def __init__(self, input_dim, data, dist_calculator):
        self.input_dim = input_dim
        self.batch_size = FLAGS.batch_size * (1 + FLAGS.num_neg)
        # Create placeholders.
        self.laplacians_1, self.laplacians_2, self.features_1, self.features_2, \
        self.num_nonzero_1, self.num_nonzero_2, self.dropout, \
        self.val_test_laplacians_1, self.val_test_laplacians_2, \
        self.val_test_features_1, self.val_test_features_2, \
        self.val_test_num_nonzero_1, self.val_test_num_nonzero_2 = \
            self._create_basic_placeholders(FLAGS.batch_size, self.batch_size,
                                            level=get_coarsen_level())

        # self.pos_interact_score = []

        # Build the model.
        super(SiameseRankingModel, self).__init__()
        self.origin_gs, self.pos_gs, self.neg_gs = self._load_pos_neg_train_pairs(
            data, dist_calculator)

    # Assume: 1.if not enough negative pairs ignore 2.too many negative pairs rand select
    def _pn_pair_generator(self, dist_mat):
        origin_gs, pos_gs, neg_gs = [], [], []
        dist_mat_idx = np.argsort(dist_mat, axis=1)
        dist_mat.sort(axis=1)
        assert (FLAGS.top_k >= 0)
        top = dist_mat.shape[1] if FLAGS.top_k == 0 else FLAGS.top_k
        for row in range(dist_mat.shape[0]):
            # pos_thresh = dist_mat[row][0] + FLAGS.pos_thresh
            for col in range(top):  # dist_mat.shape[1]
                # if dist_mat[row][col] >= pos_thresh:
                #     break
                neg_thresh = dist_mat[row][col] + FLAGS.delta
                for i in range(col + 1, dist_mat.shape[1]):
                    if dist_mat[row][i] > neg_thresh:
                        origin_gs.append(row)
                        pos_gs.append(dist_mat_idx[row][col])
                        # end = dist_mat.shape[1] if i + 100 > dist_mat.shape[1] else i + 100
                        # neg_idx = list(range(i, end))
                        neg_idx = list(range(i, dist_mat.shape[1]))
                        # TODO: sample coverage diverse?
                        # TODO: head 10 or tail 10 are more important?
                        # TODO: only top 10 PN pair (10==inter@k) / based on delta?
                        # --------------------------------------------------------
                        # TODO: ASK Ting: weight the top 10 and tail ones
                        # TODO: neural information retrival & learning to rank --good paper reading!!!! exp!!!!
                        neg_idx = SelfShuffleList([dist_mat_idx[row][idx] for idx in neg_idx])
                        neg_gs.append(neg_idx)
                        break
        return origin_gs, pos_gs, neg_gs

    def _load_pos_neg_train_pairs(self, data, dist_calculator):
        gs = [g.nxgraph for g in data.train_gs]
        dist_mat = get_gs_dist_mat(
            gs, gs, dist_calculator, 'train', 'train',
            FLAGS.dataset, FLAGS.dist_metric, FLAGS.dist_algo, FLAGS.dist_norm)
        origin_gs, pos_gs, neg_gs = self._pn_pair_generator(dist_mat)
        return SelfShuffleList(origin_gs), SelfShuffleList(pos_gs), SelfShuffleList(neg_gs)

    def pred_sim_without_act(self):
        return self.val_test_pred_score

    def apply_final_act_np(self, score):
        return score

    def _sample_train_pair(self, data):
        origin_list, pn_list = [], []
        origin_list.append(data.train_gs[self.origin_gs.get_next()])
        pn_list.append(data.train_gs[self.pos_gs.get_next()])
        neg_list = self.neg_gs.get_next()
        for _ in range(FLAGS.num_neg):
            pn_list.append(data.train_gs[neg_list.get_next()])
        return origin_list, pn_list

    def get_feed_dict_for_train(self, data):
        rtn = {}
        origin_gs, pn_gs = [], []
        for i in range(FLAGS.batch_size):
            origin_list, pn_list = self._sample_train_pair(data)
            origin_gs += origin_list
            pn_gs += pn_list
        rtn[self.dropout] = FLAGS.dropout
        return self._supply_laplacians_etc_to_feed_dict(rtn, origin_gs, pn_gs, 'train')

    def get_feed_dict_for_val_test(self, g1, g2, _):
        rtn = {}
        origin_gs, pn_gs = [g1], [g2]
        return self._supply_laplacians_etc_to_feed_dict(rtn, origin_gs, pn_gs, 'val_test')

    def get_true_sim(self, i, j, true_result):
        assert (true_result.dist_or_sim() == 'dist')
        _, d = true_result.dist_sim(i, j, FLAGS.dist_norm)
        return d

    def get_eval_metrics_for_val(self):
        return ['kendalls_tau']

    def get_eval_metrics_for_test(self):
        return ['prec@k', 'mrr', 'kendalls_tau', 'spearmans_rho', 'time']

    def _get_determining_result_for_val(self):
        return 'val_kendalls_tau'

    def _get_ins(self, layer, tvt):
        assert (layer.__class__.__name__ == 'GraphConvolution')
        ins = []
        for features in (self._get_plhdr('features_1', tvt) +
                         self._get_plhdr('features_2', tvt)):
            ins.append(features)
        return ins

    def _proc_ins_for_merging_layer(self, ins, tvt):
        proc_ins = []
        if tvt == 'train':
            pn_unit = FLAGS.num_neg + 1
            assert (len(ins) % (pn_unit + 1) == 0)
            ins_origin = ins[:FLAGS.batch_size]
            ins_pn = ins[FLAGS.batch_size:]
            ins_pos = ins_pn[0::pn_unit]
            ins_neg = ins_pn[1::pn_unit]
            for o, o_p in zip(ins_origin, ins_pos):
                proc_ins.append([o, o_p])
            for i in range(2, pn_unit):
                ins_neg += ins_pn[i::pn_unit]
                ins_origin += ins[:FLAGS.batch_size]
            assert (len(ins_neg) == len(ins_origin))
            for o, o_n in zip(ins_origin, ins_neg):
                proc_ins.append([o, o_n])
        else:
            assert (len(ins) == 2)
            proc_ins.append([ins[0], ins[1]])
        ins = proc_ins
        return ins

    def _val_test_pred_score(self):
        assert (self.val_test_output.get_shape().as_list() == [1, 1])
        pred_score = self.val_test_output[0][0]
        return tf.squeeze(pred_score)

    def _rank_loss(self, y_pred, loss_type):
        pos_interact_score = tf.squeeze(y_pred[:FLAGS.batch_size])
        neg_interact_score = tf.squeeze(y_pred[FLAGS.batch_size:])
        if loss_type == 'log_loss':
            diff_mat = tf.reshape(tf.tile(pos_interact_score, [FLAGS.num_neg]), (-1, 1)) \
                       - tf.reshape(neg_interact_score, (-1, 1))
            rank_loss = tf.reduce_mean(-tf.log(tf.sigmoid(FLAGS.gamma * diff_mat)))
        elif loss_type == 'max_margin':
            diff_mat = tf.reshape(tf.tile(pos_interact_score, [FLAGS.num_neg]), (-1, 1)) \
                       - tf.reshape(neg_interact_score, (-1, 1))
            # self.pos_interact_score += pos_interact_score
            rank_loss = tf.reduce_mean(tf.nn.relu(FLAGS.gamma - diff_mat))
        else:
            raise NotImplementedError
        return rank_loss

    def _task_loss(self, tvt):
        if tvt == 'train':
            rank_loss = self._rank_loss(self.train_outputs, 'max_margin')  # 'max_margin', 'log_loss'
        else:
            rank_loss = 0.0
        return rank_loss, 'rank loss'

    def _create_basic_placeholders(self, num1, num2, level):
        laplacians_1 = \
            [[[tf.sparse_placeholder(tf.float32)]
              for _ in range(level)] for _ in range(num1)]
        laplacians_2 = \
            [[[tf.sparse_placeholder(tf.float32)]
              for _ in range(level)] for _ in range(num2)]
        features_1 = \
            [tf.sparse_placeholder(tf.float32) for _ in range(num1)]
        features_2 = \
            [tf.sparse_placeholder(tf.float32) for _ in range(num2)]
        num_nonzero_1 = \
            [tf.placeholder(tf.int32) for _ in range(num1)]
        num_nonzero_2 = \
            [tf.placeholder(tf.int32) for _ in range(num2)]
        dropout = tf.placeholder_with_default(0., shape=())
        val_test_laplacians_1 = [[[tf.sparse_placeholder(tf.float32)] for _ in range(level)]]
        val_test_laplacians_2 = [[[tf.sparse_placeholder(tf.float32)] for _ in range(level)]]
        val_test_features_1 = [tf.sparse_placeholder(tf.float32)]
        val_test_features_2 = [tf.sparse_placeholder(tf.float32)]
        val_test_num_nonzero_1 = [tf.placeholder(tf.int32)]
        val_test_num_nonzero_2 = [tf.placeholder(tf.int32)]
        return laplacians_1, laplacians_2, features_1, features_2, \
               num_nonzero_1, num_nonzero_2, dropout, \
               val_test_laplacians_1, val_test_laplacians_2, \
               val_test_features_1, val_test_features_2, \
               val_test_num_nonzero_1, val_test_num_nonzero_2

    def _supply_laplacians_etc_to_feed_dict(self, feed_dict, origin_gs, pn_gs, tvt):
        num_laplacians = 1
        for i, origin_g in enumerate(origin_gs):
            feed_dict[self._get_plhdr('features_1', tvt)[i]] = \
                origin_g.get_node_inputs()
            feed_dict[self._get_plhdr('num_nonzero_1', tvt)[i]] = \
                origin_g.get_node_inputs_num_nonzero()
            for j in range(get_coarsen_level()):
                for k in range(num_laplacians):
                    feed_dict[self._get_plhdr('laplacians_1', tvt)[i][j][k]] = \
                        origin_g.get_laplacians(j)[k]

        for i, pn_g in enumerate(pn_gs):
            feed_dict[self._get_plhdr('features_2', tvt)[i]] = \
                pn_g.get_node_inputs()
            feed_dict[self._get_plhdr('num_nonzero_2', tvt)[i]] = \
                pn_g.get_node_inputs_num_nonzero()
            for j in range(get_coarsen_level()):
                for k in range(num_laplacians):
                    feed_dict[self._get_plhdr('laplacians_2', tvt)[i][j][k]] = \
                        pn_g.get_laplacians(j)[k]

        return feed_dict
