from config import FLAGS
import sys
from os.path import dirname, abspath
import tensorflow as tf

cur_folder = dirname(abspath(__file__))
sys.path.insert(0, '{}/../../src'.format(cur_folder))

from utils import sorted_nicely, get_ts


def solve_parent_dir():
    pass


def check_flags():
    assert (0 < FLAGS.valid_percentage < 1)
    assert (FLAGS.layer_num >= 2)
    assert (FLAGS.batch_size >= 1)
    assert (FLAGS.iters >= 1)
    assert (FLAGS.iters_val_start >= 1)
    assert (FLAGS.iters_val_every >= 1)
    assert (FLAGS.gpu >= -1)
    d = FLAGS.flag_values_dict()
    ln = d['layer_num']
    ls = [False] * ln
    for k in d.keys():
        if 'layer_' in k:
            lt = k.split('_')[1]
            if lt != 'num':
                i = int(lt) - 1
                if not (0 <= i < len(ls)):
                    raise RuntimeError('Wrong spec {}'.format(k))
                ls[i] = True
    for i, x in enumerate(ls):
        if not x:
            raise RuntimeError('layer {} not specified'.format(i + 1))
    # TODO: finish.


def get_flags(k):
    if hasattr(FLAGS, k):
        return getattr(FLAGS, k)
    else:
        return None


def convert_msec_to_sec_str(sec):
    return '{:.2f}msec'.format(sec * 1000)


def convert_long_time_to_str(sec):
    day = sec // (24 * 3600)
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    minutes = sec // 60
    sec %= 60
    seconds = sec
    return '{} days {} hours {} mins {:.1f} secs'.format(
        int(day), int(hour), int(minutes), seconds)


def get_siamese_dir():
    return cur_folder


def get_coarsen_level():
    if FLAGS.coarsening:
        return int(FLAGS.coarsening[6:])
    else:
        return 1

def get_model_info_as_str(model_info_table=None):
    rtn = []
    d = FLAGS.flag_values_dict()
    for k in sorted_nicely(d.keys()):
        v = d[k]
        s = '{0:26} : {1}'.format(k, v)
        rtn.append(s)
        if model_info_table:
            model_info_table.append([k, '**{}**'.format(v)])
    rtn.append('{0:26} : {1}'.format('ts', get_ts()))
    return '\n'.join(rtn)


def need_val(iter):
    assert (iter != 0)  # 1-based iter
    return iter == 1 or \
           (iter >= FLAGS.iters_val_start and iter % FLAGS.iters_val_every == 0)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res
