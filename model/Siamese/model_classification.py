from config import FLAGS
from model import Model
from samplers import SelfShuffleList
from utils_siamese import get_coarsen_level
from classification import get_classification_labels_from_dist_mat, classify
from dist_calculator import get_gs_dist_mat
import numpy as np
import tensorflow as tf


class SiameseClassificationModel(Model):
    def __init__(self, input_dim, data, dist_calculator):
        self.input_dim = input_dim
        print('original_input_dim', self.input_dim)
        self.num_class = 2
        self.laplacians_1, self.laplacians_2, self.features_1, self.features_2, \
        self.num_nonzero_1, self.num_nonzero_2, self.dropout, \
        self.val_test_laplacians_1, self.val_test_laplacians_2, \
        self.val_test_features_1, self.val_test_features_2, \
        self.val_test_num_nonzero_1, self.val_test_num_nonzero_2 = \
            self._create_basic_placeholders(FLAGS.batch_size, FLAGS.batch_size,
                                            level=get_coarsen_level())
        self.train_y_true = tf.placeholder(
            tf.float32, shape=(FLAGS.batch_size, self.num_class))
        self.val_test_y_true = tf.placeholder(
            tf.float32, shape=(1, self.num_class))
        # Build the model.
        super(SiameseClassificationModel, self).__init__()
        self.pos_pairs, self.neg_pairs = self._load_pos_neg_train_pairs(
            data, dist_calculator)
        self.cur_sample_class = 1  # 1 for pos, -1 for neg

    def pred_sim_without_act(self):
        return self.val_test_pred_score

    def apply_final_act_np(self, score):
        # Transform the prediction score into classification score.
        assert (0 <= score <= 1)
        # if score >= 0.5:
        #     return float('inf')  # similar pair
        # else:
        #     return -float('inf')  # dissimilar pair
        return score

    def get_feed_dict_for_train(self, data):
        rtn = {}
        pairs = []
        y_true = np.zeros((FLAGS.batch_size, self.num_class))
        for i in range(FLAGS.batch_size):
            g1, g2, label = self._sample_train_pair(data)
            pairs.append((g1, g2))
            y_true[i] = label
        rtn[self.train_y_true] = y_true
        rtn[self.dropout] = FLAGS.dropout
        return self._supply_laplacians_etc_to_feed_dict(rtn, pairs, 'train')

    def get_feed_dict_for_val_test(self, g1, g2, true_sim):
        rtn = {}
        pairs = [(g1, g2)]
        y_true = np.zeros((1, self.num_class))
        y_true[0] = self._class_to_one_hot_label(true_sim)
        rtn[self.val_test_y_true] = y_true
        return self._supply_laplacians_etc_to_feed_dict(rtn, pairs, 'val_test')

    def get_true_sim(self, i, j, true_result):
        assert (true_result.dist_or_sim() == 'dist')
        _, d = true_result.dist_sim(i, j, FLAGS.dist_norm)
        c = classify(d, FLAGS.thresh_val_test_pos, FLAGS.thresh_val_test_neg)
        if c != 0:
            return c
        else:
            return None

    def get_eval_metrics_for_val(self):
        return ['loss', 'acc']

    def get_eval_metrics_for_test(self):
        return ['acc', 'pos_acc', 'neg_acc', 'prec@k', 'mrr', 'auc',
                'time', 'emb_vis_binary', 'emb_vis_gradual', 'ranking',
                'attention', 'draw_heat_hist', 'draw_gt_rk']

    def _get_determining_result_for_val(self):
        return 'val_acc'

    def _val_need_max(self):
        return True

    def _get_ins(self, layer, tvt):
        assert (layer.__class__.__name__ == 'GraphConvolution' or
                layer.__class__.__name__ == 'GraphConvolutionAttention')
        ins = []
        for features in (self._get_plhdr('features_1', tvt) +
                         self._get_plhdr('features_2', tvt)):
            ins.append(features)
        return ins

    def _proc_ins_for_merging_layer(self, ins, _):
        assert (len(ins) % 2 == 0)
        proc_ins = []
        i = 0
        j = len(ins) // 2
        for _ in range(len(ins) // 2):
            proc_ins.append([ins[i], ins[j]])
            i += 1
            j += 1
        ins = proc_ins
        return ins

    def _val_test_pred_score(self):
        # assert (self.val_test_output.get_shape().as_list()[0] == 1)
        pred_score = tf.nn.softmax(self.val_test_output)
        # assert (pred_score.get_shape().as_list() == [1, 2])
        pred_score = pred_score[0][0]
        return tf.squeeze(pred_score)

    def _task_loss(self, tvt):
        if tvt == 'train':
            y_pred = self._stack_concat(self.train_outputs)
            y_true = self.train_y_true
        else:
            y_pred = self._stack_concat(self.val_test_output)
            y_true = self.val_test_y_true
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=y_true, logits=y_pred)), \
               'cross_entropy_loss'

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

    def _load_pos_neg_train_pairs(self, data, dist_calculator):
        gs = [g.nxgraph for g in data.train_gs]
        dist_mat = get_gs_dist_mat(
            gs, gs, dist_calculator, 'train', 'train',
            FLAGS.dataset, FLAGS.dist_metric, FLAGS.dist_algo, FLAGS.dist_norm)
        _, _, _, pos_pairs, neg_pairs = \
            get_classification_labels_from_dist_mat(
                dist_mat, FLAGS.thresh_train_pos, FLAGS.thresh_train_neg)
        return SelfShuffleList(pos_pairs), SelfShuffleList(neg_pairs)

    def _sample_train_pair(self, data):
        if self.cur_sample_class == 1:
            li = self.pos_pairs
            label = self._class_to_one_hot_label(self.cur_sample_class)
            self.cur_sample_class = -1
        elif self.cur_sample_class == -1:
            li = self.neg_pairs
            label = self._class_to_one_hot_label(self.cur_sample_class)
            self.cur_sample_class = 1
        else:
            assert (False)
        x, y = li.get_next()
        # print(x, y, label)
        return data.train_gs[x], data.train_gs[y], label

    def _class_to_one_hot_label(self, c):
        if c == 1:
            return [1, 0]
        elif c == -1:
            return [0, 1]
        else:
            assert (False)

    def _supply_laplacians_etc_to_feed_dict(self, feed_dict, pairs, tvt):
        for i, (g1, g2) in enumerate(pairs):
            feed_dict[self._get_plhdr('features_1', tvt)[i]] = \
                g1.get_node_inputs()
            feed_dict[self._get_plhdr('features_2', tvt)[i]] = \
                g2.get_node_inputs()
            feed_dict[self._get_plhdr('num_nonzero_1', tvt)[i]] = \
                g1.get_node_inputs_num_nonzero()
            feed_dict[self._get_plhdr('num_nonzero_2', tvt)[i]] = \
                g2.get_node_inputs_num_nonzero()
            num_laplacians = 1
            for j in range(get_coarsen_level()):
                for k in range(num_laplacians):
                    feed_dict[self._get_plhdr('laplacians_1', tvt)[i][j][k]] = \
                        g1.get_laplacians(j)[k]
                    feed_dict[self._get_plhdr('laplacians_2', tvt)[i][j][k]] = \
                        g2.get_laplacians(j)[k]
        return feed_dict
