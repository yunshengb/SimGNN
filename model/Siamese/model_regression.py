from config import FLAGS
from model import Model
from samplers import SelfShuffleList
from utils_siamese import get_coarsen_level, get_flags
from similarity import create_sim_kernel
from dist_calculator import get_gs_dist_mat
import numpy as np
import tensorflow as tf
from fake_generator import graph_generator
from data_siamese import ModelGraph
from utils import save, load, get_save_path


class SiameseRegressionModel(Model):
    def __init__(self, input_dim, data, dist_calculator):
        self.input_dim = input_dim
        print('original_input_dim', self.input_dim)
        self.laplacians_1, self.laplacians_2, self.features_1, self.features_2, \
        self.num_nonzero_1, self.num_nonzero_2, self.dropout, \
        self.val_test_laplacians_1, self.val_test_laplacians_2, \
        self.val_test_features_1, self.val_test_features_2, \
        self.val_test_num_nonzero_1, self.val_test_num_nonzero_2 = \
            self._create_basic_placeholders(FLAGS.batch_size, FLAGS.batch_size,
                                            level=get_coarsen_level())
        self.train_y_true = tf.placeholder(
            tf.float32, shape=(FLAGS.batch_size, 1))
        self.val_test_y_true = tf.placeholder(
            tf.float32, shape=(1, 1))
        # Build the model.
        super(SiameseRegressionModel, self).__init__()
        self.sim_kernel = create_sim_kernel(
            FLAGS.sim_kernel, get_flags('yeta'), get_flags('scale'))
        self.train_triples = self._load_train_triples(data, dist_calculator)

    def pred_sim_without_act(self):
        return self.val_test_pred_score

    def apply_final_act_np(self, score):
        return score

    def get_feed_dict_for_train(self, data):
        rtn = {}
        pairs = []
        y_true = np.zeros((FLAGS.batch_size, 1))
        for i in range(FLAGS.batch_size):
            g1, g2, true_sim = self._sample_train_pair(data)
            pairs.append((g1, g2))
            y_true[i] = true_sim
        rtn[self.train_y_true] = y_true
        rtn[self.dropout] = FLAGS.dropout
        return self._supply_laplacians_etc_to_feed_dict(rtn, pairs, 'train')

    def get_feed_dict_for_val_test(self, g1, g2, true_sim):
        rtn = {}
        pairs = [(g1, g2)]
        y_true = np.zeros((1, 1))
        y_true[0] = true_sim
        rtn[self.val_test_y_true] = y_true
        return self._supply_laplacians_etc_to_feed_dict(rtn, pairs, 'val_test')

    def get_true_sim(self, i, j, true_result):
        assert (true_result.dist_or_sim() == 'dist')
        _, d = true_result.dist_sim(i, j, FLAGS.dist_norm)
        true_sim = self.sim_kernel.dist_to_sim_np(d)
        return true_sim

    def get_eval_metrics_for_val(self):
        return ['loss', 'mse']

    def get_eval_metrics_for_test(self):
        return ['mse', 'prec@k', 'prec@k_0.005', 'mrr', 'kendalls_tau', 'spearmans_rho', 'time',
                'groundtruth', 'ranking', 'attention', 'emb_vis_gradual', 'draw_heat_hist',
                'draw_gt_rk']

    def _get_determining_result_for_val(self):
        return 'val_mse'

    def _val_need_max(self):
        return False

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
        assert (self.val_test_output.get_shape().as_list() == [1, 1])
        return tf.squeeze(self.val_test_output)

    def _task_loss(self, tvt):
        if tvt == 'train':
            y_pred = self._stack_concat(self.train_outputs)
            y_true = self.train_y_true
        else:
            y_pred = self._stack_concat(self.val_test_output)
            y_true = self.val_test_y_true
        assert (y_true.get_shape() == y_pred.get_shape())
        loss = tf.reduce_mean(tf.nn.l2_loss(y_true - y_pred))
        return loss, 'mse_loss'

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

    def _load_train_triples(self, data, dist_calculator):
        gs = [g.nxgraph for g in data.train_gs]
        dist_mat = get_gs_dist_mat(
            gs, gs, dist_calculator, 'train', 'train',
            FLAGS.dataset, FLAGS.dist_metric, FLAGS.dist_algo, FLAGS.dist_norm)
        m, n = dist_mat.shape
        triples = []

        generate_flag = FLAGS.fake_generation is not None
        repeat_flag = FLAGS.top_repeater is not None
        if generate_flag:
            assert ('fake_' in FLAGS.fake_generation)
            assert (not FLAGS.top_repeater)
            fake_num = int(FLAGS.fake_generation.split('_')[1])
            filepath = get_save_path() + '/{}_fake_{}'.format(
                FLAGS.dataset, fake_num)
            load_data = load(filepath)
            if load_data:
                print('Loaded from {} with {} triples'.format(
                    filepath, len(load_data.li)))
                return load_data
            node_feat_encoder = data.node_feat_encoder
        elif repeat_flag:
            assert ('_repeat_' in FLAGS.top_repeater)
            assert (not FLAGS.fake_generation)
            top_num = int(FLAGS.top_repeater.split('_')[0])
            repeat_num = int(FLAGS.top_repeater.split('_')[2])

        dist_mat_idx = np.argsort(dist_mat, axis=1)
        for i in range(m):
            g1 = data.train_gs[i]
            if generate_flag:
                sample_graphs, sample_geds = graph_generator(g1.nxgraph, fake_num)
                print(i, m, sample_geds)
                for sample_g, sample_ged in zip(sample_graphs, sample_geds):
                    triples.append((ModelGraph(g1.nxgraph, node_feat_encoder),
                                    ModelGraph(sample_g, node_feat_encoder),
                                    self.sim_kernel.dist_to_sim_np(sample_ged)))
            for j in range(n):
                col = dist_mat_idx[i][j]
                g2, ged = data.train_gs[col], dist_mat[i][col]
                triples.append((g1, g2, self.sim_kernel.dist_to_sim_np(ged)))
                if repeat_flag and j <= top_num:
                    for _ in range(repeat_num):
                        triples.append((g1, g2, self.sim_kernel.dist_to_sim_np(ged)))
        rtn = SelfShuffleList(triples)
        if generate_flag:
            save(filepath, rtn)
            print('Saved to {} with {} triples'.format(filepath, len(rtn.li)))
        return rtn

    def _sample_train_pair(self, data):
        x, y, true_sim = self.train_triples.get_next()
        # print(x, y, true_sim)
        # return data.train_gs[x], data.train_gs[y], true_sim
        return x, y, true_sim

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
