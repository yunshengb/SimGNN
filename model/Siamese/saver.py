from config import FLAGS
from utils import get_ts, create_dir_if_not_exists, save_as_dict
from utils_siamese import get_siamese_dir, get_model_info_as_str, need_val
import tensorflow as tf
from glob import glob
from os import system


class Saver(object):
    def __init__(self, sess):
        model_str = self._get_model_str()
        self.logdir = '{}/logs/{}_{}'.format(
            get_siamese_dir(), model_str, get_ts())
        create_dir_if_not_exists(self.logdir)
        self.tw = tf.summary.FileWriter(self.logdir + '/train', sess.graph)
        self.vw = tf.summary.FileWriter(self.logdir + '/val', sess.graph)
        self.all_merged = tf.summary.merge_all()
        self.loss_merged = tf.summary.merge(
            self._extract_loss_related_summaries_as_list())
        self._log_model_info(self.logdir, sess)
        self.f = open('{}/results_{}.txt'.format(self.logdir, get_ts()), 'w')
        print('Logging to {}'.format(self.logdir))

    def get_log_dir(self):
        return self.logdir

    def proc_objs(self, objs, tvt, iter):
        if tvt == 'train':
            # Only loss the loss-related summaries when not validating.
            objs.insert(0, self.all_merged if need_val(iter)
            else self.loss_merged)
        return objs

    def proc_outs(self, outs, tvt, iter):
        if tvt == 'train':
            self.tw.add_summary(outs[0], iter)


    def log_val_results(self, results, iter):
        for metric, num in results.items():
            if metric == 'val_loss':
                metric = 'total_loss'
            summary = tf.Summary(
                value=[tf.Summary.Value(tag=metric, simple_value=num)])
            self.vw.add_summary(summary, iter)

    def save_train_val_info(self, train_costs, train_times,
                            val_results_dict):
        sfn = '{}/train_val_info'.format(self.logdir)
        flags = FLAGS.flag_values_dict()
        ts = get_ts()
        save_as_dict(sfn, train_costs, train_times, val_results_dict, flags, ts)

    def save_test_info(self, sim_mat, time_li, best_iter, test_results,
                       node_embs_list, graph_embs_mat, emb_time, atts):
        self._save_to_result_file(best_iter, 'best iter')
        self._save_to_result_file(test_results, 'test results')
        sfn = '{}/test_info'.format(self.logdir)
        # The following function call must be made in one line!
        save_as_dict(sfn, sim_mat, time_li, best_iter, test_results, node_embs_list, graph_embs_mat, emb_time, atts)

    def save_overall_time(self, overall_time):
        self._save_to_result_file(overall_time, 'overall time')

    def clean_up_saved_models(self, best_iter):
        for file in glob('{}/models/*'.format(self.get_log_dir())):
            if str(best_iter) not in file:
                system('rm -rf {}'.format(file))

    def _get_model_str(self):
        li = []
        for f in [FLAGS.model, FLAGS.dataset]:
            li.append(str(f))
        return '_'.join(li)

    def _log_model_info(self, logdir, sess):
        model_info_table = [["**key**", "**value**"]]
        with open(logdir + '/model_info.txt', 'w') as f:
            s = get_model_info_as_str(model_info_table)
            f.write(s)
        model_info_op = \
            tf.summary.text(
                'model_info', tf.convert_to_tensor(model_info_table))
        self.tw.add_summary(sess.run(model_info_op))

    def _save_to_result_file(self, obj, name):
        self.f.write('{}: {}\n'.format(name, obj))

    def _extract_loss_related_summaries_as_list(self):
        rtn = []
        for tensor in tf.get_collection(tf.GraphKeys.SUMMARIES):
            # Assume "loss" is in the loss-related summary tensors.
            if 'loss' in tensor.name:
                rtn.append([tensor])
        return rtn

    def _bool_to_str(self, b, s):
        assert (type(b) is bool)
        if b:
            return s
        else:
            return 'NO{}'.format(s)
