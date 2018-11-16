from config import FLAGS
from layers_factory import create_layers
import numpy as np
import tensorflow as tf
from warnings import warn


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        self.vars = {}
        self.layers = []
        self.train_loss = 0
        self.val_test_loss = 0
        self.optimizer = None
        self.opt_op = None

        self.batch_size = FLAGS.batch_size
        self.weight_decay = FLAGS.weight_decay
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)

        self._build()
        print('Flow built')
        # Build metrics
        self._loss()
        print('Loss built')
        self.opt_op = self.optimizer.minimize(self.train_loss)
        print('Optimizer built')

    def _build(self):
        # Create layers according to FLAGS.
        self.layers = create_layers(self)
        assert (len(self.layers) > 0)
        print('Created {} layers: {}'.format(
            len(self.layers), ', '.join(l.get_name() for l in self.layers)))

        # Build the siamese model for train and val_test, respectively,
        for tvt in ['train', 'val_test']:
            print(tvt)
            # Go through each layer except the last one.
            acts = [self._get_ins(self.layers[0], tvt)]
            outs = None
            for k, layer in enumerate(self.layers):
                ins = self._proc_ins(acts[-1], k, layer, tvt)
                print(layer.name)
                outs = layer(ins)
                outs = self._proc_outs(outs, k, layer, tvt)
                acts.append(outs)
            if tvt == 'train':
                self.train_outputs = outs
                self.train_acts = acts
            else:
                self.val_test_output = outs
                self.val_test_pred_score = self._val_test_pred_score()
                self.val_test_acts = acts

        self.node_embeddings = self._get_last_gcn_layer_outputs('val_test')

        # Store model variables for easy access.
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.vars = {var.name: var for var in variables}

    def _loss(self):
        self.train_loss = self._loss_helper('train')
        self.val_test_loss = self._loss_helper('val')

    def _loss_helper(self, tvt):
        rtn = 0

        # Weight decay loss.
        wdl = 0
        for layer in self.layers:
            for var in layer.vars.values():
                wdl = self.weight_decay * tf.nn.l2_loss(var)
                rtn += wdl
        if tvt == 'train':
            tf.summary.scalar('weight_decay_loss', wdl)

        loss, loss_label = self._task_loss(tvt)
        rtn += loss
        if tvt == 'train':
            tf.summary.scalar(loss_label, loss)

        if FLAGS.graph_loss == '1st':
            node_emb_list = self._get_last_gcn_layer_outputs(tvt)
            laplacian_list = self._get_laplacians_for_graph_loss(tvt)
            gl = 0
            for i, node_emb_mat in enumerate(node_emb_list):
                # gli = 2 * tf.trace(
                #     dot(tf.transpose(
                #         dot(laplacian_list[i], node_emb_mat, sparse=True)),
                #         node_emb_mat))
                # gl += gli
                mat = tf.matmul(node_emb_mat, tf.transpose(node_emb_mat))
                gl += tf.sqrt(tf.reduce_sum(tf.square(tf.sparse_add(-mat, laplacian_list[i][0]))))
            gl /= FLAGS.batch_size
            gl *= FLAGS.graph_loss_alpha
            rtn += gl
            if tvt == 'train':
                tf.summary.scalar('1st_order_graph_loss', gl)

        if tvt == 'train':
            tf.summary.scalar('total_loss', rtn)
        return rtn

    def pred_sim_without_act(self):
        raise NotImplementedError()

    def apply_final_act_np(self, score):
        raise NotImplementedError()

    def get_feed_dict_for_train(self, data):
        raise NotImplementedError()

    def get_feed_dict_for_val_test(self, g1, g2, true_sim):
        raise NotImplementedError()

    def get_true_sim(self, i, j, true_result):
        raise NotImplementedError()

    def get_eval_metrics_for_val(self):
        raise NotImplementedError()

    def get_eval_metrics_for_test(self):
        raise NotImplementedError()

    def _get_determining_result_for_val(self):
        raise NotImplementedError()

    def _val_need_max(self):
        raise NotImplementedError()

    def find_load_best_model(self, sess, saver, val_results_dict):
        cur_max_metric = -float('inf')
        cur_min_metric = float('inf')
        cur_best_iter = 1
        metric_list = []
        early_thresh = int(FLAGS.iters * 0.1)
        deter_r_name = self._get_determining_result_for_val()
        for iter, val_results in val_results_dict.items():
            metric = val_results[deter_r_name]
            metric_list.append(metric)
            if iter >= early_thresh:
                if self._val_need_max():
                    if metric >= cur_max_metric:
                        cur_max_metric = metric
                        cur_best_iter = iter
                else:
                    if metric <= cur_min_metric:
                        cur_min_metric = metric
                        cur_best_iter = iter
        if self._val_need_max():
            argfunc = np.argmax
            takefunc = np.max
            best_metric = cur_max_metric
        else:
            argfunc = np.argmin
            takefunc = np.min
            best_metric = cur_min_metric
        global_best_iter = list(val_results_dict.items()) \
            [int(argfunc(metric_list))][0]
        global_best_metirc = takefunc(metric_list)
        if global_best_iter != cur_best_iter:
            warn(
                'The global best iter is {} with {}={:.5f},\nbut the '
                'best iter after first 10% iterations is {} with {}={:.5f}'.format(
                    global_best_iter, deter_r_name, global_best_metirc,
                    cur_best_iter, deter_r_name, best_metric))
        lp = '{}/models/{}.ckpt'.format(saver.get_log_dir(), cur_best_iter)
        self.load(sess, lp)
        print('Loaded the best model at iter {} with {} {:.5f}'.format(
            cur_best_iter, deter_r_name, best_metric))
        return cur_best_iter
        # return None

    def _get_ins(self, layer, tvt):
        raise NotImplementedError()

    def _supply_laplacians_etc_to_ins(self, ins, tvt, gcn_id):
        if not FLAGS.coarsening:
            gcn_id = 0
        for i, (laplacians, num_nonzero) in \
                enumerate(zip(
                    self._get_plhdr('laplacians_1', tvt) +
                    self._get_plhdr('laplacians_2', tvt),
                    self._get_plhdr('num_nonzero_1', tvt) +
                    self._get_plhdr('num_nonzero_2', tvt))):
            ins[i] = [ins[i], laplacians[gcn_id], num_nonzero]
        return ins

    def _proc_ins_for_merging_layer(self, ins, tvt):
        raise NotImplementedError()

    def _val_test_pred_score(self):
        raise NotImplementedError()

    def _task_loss(self, tvt):
        raise NotImplementedError()

    def _proc_ins(self, ins, k, layer, tvt):
        ln = layer.__class__.__name__
        ins_mat = None
        if k != 0 and tvt == 'train':
            # sparse matrices (k == 0; the first layer) cannot be logged.
            need_log = True
        else:
            need_log = False
        if ln == 'GraphConvolution' or ln == 'GraphConvolutionAttention':
            gcn_count = int(layer.name.split('_')[-1])
            assert (gcn_count >= 1)  # 1-based
            gcn_id = gcn_count - 1
            ins = self._supply_laplacians_etc_to_ins(ins, tvt, gcn_id)
            if need_log:
                ins_mat = self._stack_concat([i[0] for i in ins])
        else:
            ins_mat = self._stack_concat(ins)
            if layer.merge_graph_level_embs():
                ins = self._proc_ins_for_merging_layer(ins, tvt)
            if ln == 'Dense':
                # Use matrix operations instead of iterating through list
                # after the merging layer.
                ins = ins_mat
        if need_log:
            self._log_mat(ins_mat, layer, 'ins')
        return ins

    def _proc_outs(self, outs, k, layer, tvt):
        outs_mat = self._stack_concat(outs)
        ln = layer.__class__.__name__
        if tvt == 'train':
            self._log_mat(outs_mat, layer, 'outs')
        if tvt == 'val_test' and layer.produce_graph_level_emb() and \
                not FLAGS.coarsening:
            if ln != 'ANPM' and ln != 'ANPMD' and ln != 'ANNH':
                embs = outs
            else:
                embs = layer.embeddings
            assert (len(embs) == 2)
            # Note: some architecture may NOT produce
            # any graph-level embeddings.
            self.graph_embeddings = embs
            s = embs[0].get_shape().as_list()
            assert (s[0] == 1)
            self.embed_dim = s[1]
        if tvt == 'val_test' and layer.produce_node_atts():
            if ln == 'Attention':
                assert (len(outs) == 2)
            self.attentions = layer.att
            s = self.attentions.get_shape().as_list()
            assert (s[1] == 1)
        return outs

    def _get_plhdr(self, key, tvt):
        if tvt == 'train':
            return self.__dict__[key]
        else:
            assert (tvt == 'test' or tvt == 'val' or tvt == 'val_test')
            return self.__dict__['val_test_' + key]

    def _get_last_gcn_layer_outputs(self, tvt):
        acts = self.train_acts if tvt == 'train' else self.val_test_acts
        assert (len(acts) == len(self.layers) + 1)
        idx = None
        for k, layer in enumerate(self.layers):
            if 'GraphConvolution' not in layer.__class__.__name__:
                idx = k
                break
        assert (idx)
        return acts[idx]

    def _get_laplacians_for_graph_loss(self, tvt):
        rtn = []
        for laplacians in (self._get_plhdr('laplacians_1', tvt) +
                           self._get_plhdr('laplacians_2', tvt)):
            assert (len(laplacians) == 1)
            rtn.append(laplacians[0])
        return rtn

    def _stack_concat(self, x):
        if type(x) is list:
            list_of_tensors = x
            assert (list_of_tensors)
            s = list_of_tensors[0].get_shape()
            if s != ():
                return tf.concat(list_of_tensors, 0)
            else:
                return tf.stack(list_of_tensors)
        else:
            # assert(len(x.get_shape()) == 2) # should be a 2-D matrix
            return x

    def _log_mat(self, mat, layer, label):
        tf.summary.histogram(layer.name + '/' + label, mat)

    def save(self, sess, saver, iter):
        logdir = saver.get_log_dir()
        sp = '{}/models/{}.ckpt'.format(logdir, iter)
        tf.train.Saver(self.vars).save(sess, sp)

    def load(self, sess, load_path):
        tf.train.Saver(self.vars).restore(sess, load_path)
