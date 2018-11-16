from utils_siamese import get_siamese_dir, get_model_info_as_str
from utils import get_ts
from main import main
from config import FLAGS
import numpy as np
import tensorflow as tf

dataset = 'aids700nef'
file_name = '{}/logs/parameter_tuning_{}_{}.csv'.format(
    get_siamese_dir(), dataset, get_ts())

header = ['valid_percentage', 'node_feat_name', 'node_feat_encoder',
          'dist_metric', 'dist_algo', 'sampler', 'sample_num', 'sampler_duplicate_removal',
          'batch_size', 'dist_norm', 'pos_thresh', 'neg_thresh',
          'graph_loss', 'graph_loss_alpha', 'dropout', 'weight_decay', 'learning_rate', 'iters', 'iters_val',
          'plot_results',
          'best_train_loss', 'best_train_iter', 'best_val_loss', 'best_val_iter',
          'train_time', 'acc_norm', 'time_norm']
# header = sorted(header)
model = 'siamese_classification'
batch_size_range = [32, 64, 128]
pn_thresh_range = [[0.63, 1.63], [0.95, 0.95]]
graph_loss_range = [None]  # what is the other option?
dropout_range = [0]
lr_range = [0.01, 0.05]


def tune_structure(FLAGS, structure_info):
    f = setup_file()
    csv_record(header + ['structure_info'], f)
    train_costs, train_times, val_results_dict, best_iter, test_results \
        = main()

    best_train_loss = np.min(train_costs)
    best_train_iter = np.argmin(train_costs)
    best_val_loss = val_results_dict[best_iter]['val_loss']
    best_val_iter = best_iter
    train_time = sum(train_times)
    print('best_train_loss: {}'.format(best_train_loss),
          'best_val_loss: {}'.format(best_val_loss),
          'best_train_iter: {}'.format(best_train_iter),
          'best_val_iter: {}'.format(best_val_iter),
          'train_time: {}'.format(train_time))

    model_results = parse_results(test_results)
    mesg = [str(x) for x in
            [FLAGS.valid_percentage,
             FLAGS.node_feat_name,
             FLAGS.node_feat_encoder,
             FLAGS.dist_metric,
             FLAGS.dist_algo,
             FLAGS.sampler,
             FLAGS.sample_num,
             FLAGS.sampler_duplicate_removal,
             FLAGS.batch_size,
             FLAGS.dist_norm,
             FLAGS.pos_thresh,
             FLAGS.neg_thresh,
             FLAGS.graph_loss,
             FLAGS.graph_loss_alpha,
             FLAGS.dropout,
             FLAGS.weight_decay,
             FLAGS.learning_rate,
             FLAGS.iters,
             FLAGS.iters_val,
             FLAGS.plot_results,
             best_train_loss,
             best_train_iter,
             best_val_loss,
             best_val_iter,
             train_time] + model_results]
    mesg.append(structure_info)
    csv_record(mesg, f)


def tune_parameter(FLAGS):
    print('Remember to clean up "../../save/SiameseModelData*" '
          'if something does not work!')
    f = setup_file()
    csv_record(header, f)

    best_parameter = [32, [0.63, 1.63], None, 0, 0.001]
    ranges = [batch_size_range, pn_thresh_range, graph_loss_range, dropout_range, lr_range]
    i = 1
    for idx, range in enumerate(ranges):
        best_results_train_loss, best_results_val_loss = float('Inf'), float('Inf')
        results_train = []
        results_val = []
        best_idx = 0
        for index, val in enumerate(range):
            best_parameter[idx] = val
            batch_size, pn_thresh, graph_loss, dropout, learning_rate = best_parameter

            flags = tf.app.flags
            reset_flag(FLAGS, flags.DEFINE_integer, 'batch_size', batch_size)
            reset_flag(FLAGS, flags.DEFINE_float, 'thresh_train_pos', pn_thresh[0])
            reset_flag(FLAGS, flags.DEFINE_float, 'thresh_train_neg', pn_thresh[1])
            reset_flag(FLAGS, flags.DEFINE_float, 'thresh_val_test_pos', pn_thresh[0])
            reset_flag(FLAGS, flags.DEFINE_float, 'thresh_val_test_neg', pn_thresh[1])
            reset_flag(FLAGS, flags.DEFINE_string, 'graph_loss', graph_loss)
            reset_flag(FLAGS, flags.DEFINE_float, 'learning_rate', learning_rate)
            reset_flag(FLAGS, flags.DEFINE_float, 'dropout', dropout)

            print('Number of tuning iteration: {}'.format(i),
                  'batch_size: {}'.format(batch_size), 'pos_thresh: {}'.format(pn_thresh[0]),
                  'neg_thresh: {}'.format(pn_thresh[1]), 'learning_rate: {}'.format(learning_rate),
                  'graph_loss: {}'.format(graph_loss))
            i += 1

            reset_flag(FLAGS, flags.DEFINE_string, 'dataset', dataset)
            reset_flag(FLAGS, flags.DEFINE_float, 'valid_percentage', 0.25)
            reset_flag(FLAGS, flags.DEFINE_string, 'model', model)
            reset_flag(FLAGS, flags.DEFINE_integer, 'layer_num', 7)
            reset_flag(
                FLAGS, flags.DEFINE_string,
                'layer_1',
                'GraphConvolution:output_dim=64,act=relu,'
                'dropout=True,bias=True,sparse_inputs=True')
            reset_flag(
                FLAGS, flags.DEFINE_string,
                'layer_2',
                'GraphConvolution:input_dim=64,output_dim=32,act=identity,'
                'dropout=True,bias=True,sparse_inputs=False')
            reset_flag(
                FLAGS, flags.DEFINE_string,
                'layer_3',
                'GraphConvolution:input_dim=32,output_dim=16,act=identity,'
                'dropout=True,bias=True,sparse_inputs=False')
            reset_flag(
                FLAGS, flags.DEFINE_string,
                'layer_4',
                'Average')
            reset_flag(
                FLAGS, flags.DEFINE_string,
                'layer_5',
                'NTN:input_dim=16,feature_map_dim=16,inneract=relu,apply_u=False,'
                'dropout=True,bias=True')
            reset_flag(
                FLAGS, flags.DEFINE_string,
                'layer_6',
                'Dense:input_dim=16,output_dim=9,dropout=True,'
                'act=relu,bias=True')
            reset_flag(
                FLAGS, flags.DEFINE_string,
                'layer_7',
                'Dense:input_dim=9,output_dim=2,dropout=True,'
                'act=identity,bias=True')

            reset_flag(FLAGS, flags.DEFINE_bool, 'dist_norm', True)
            reset_flag(FLAGS, flags.DEFINE_bool, 'plot_results', True)
            FLAGS = tf.app.flags.FLAGS
            train_costs, train_times, val_results_dict, best_iter, test_results \
                = main()

            best_train_loss = np.min(train_costs)
            best_train_iter = np.argmin(train_costs)
            best_val_loss = val_results_dict[best_iter]['val_loss']
            best_val_iter = best_iter
            train_time = sum(train_times)
            print('best_train_loss: {}'.format(best_train_loss),
                  'best_val_loss: {}'.format(best_val_loss),
                  'best_train_iter: {}'.format(best_train_iter),
                  'best_val_iter: {}'.format(best_val_iter),
                  'train_time: {}'.format(train_time))

            model_results = parse_results(test_results)
            mesg = [str(x) for x in
                    [FLAGS.valid_percentage,
                     FLAGS.node_feat_name,
                     FLAGS.node_feat_encoder,
                     FLAGS.dist_metric,
                     FLAGS.dist_algo,
                     FLAGS.sampler,
                     FLAGS.sample_num,
                     FLAGS.sampler_duplicate_removal,
                     FLAGS.batch_size,
                     FLAGS.dist_norm,
                     FLAGS.pos_thresh,
                     FLAGS.neg_thresh,
                     FLAGS.graph_loss,
                     FLAGS.graph_loss_alpha,
                     FLAGS.dropout,
                     FLAGS.weight_decay,
                     FLAGS.learning_rate,
                     FLAGS.iters,
                     FLAGS.iters_val,
                     FLAGS.plot_results,
                     best_train_loss,
                     best_train_iter,
                     best_val_loss,
                     best_val_iter,
                     train_time] + model_results]

            csv_record(mesg, f)

            if best_train_loss < best_results_train_loss:
                best_results_train_loss = best_train_loss
                results_train = mesg

            if best_val_loss < best_results_val_loss:
                best_results_val_loss = best_val_loss
                results_val = mesg
                best_idx = index

        print(results_train)
        print(results_val)

        csv_record(['Final results (best train loss):'], f)
        csv_record(header, f)
        csv_record([str(x) for x in results_train], f)

        csv_record(['Final results (best val loss):'], f)
        csv_record(header, f)
        csv_record([str(x) for x in results_val], f)

        best_parameter[idx] = best_idx

    print(best_parameter)
    csv_record(['Best Parameters:'], f)
    csv_record([str(x) for x in best_parameter], f)


def setup_file():
    f = open(file_name, 'w')
    f.write(get_model_info_as_str())
    f.write(','.join(map(str, header)) + '\n')
    f.flush()
    return f


def parse_results(results):
    acc_norm = results['acc_norm'][model]
    time_norm = results['time_norm'][model]
    model_results = [acc_norm, time_norm]
    return model_results


def csv_record(mesg, f):
    f.write(','.join(map(str, mesg)) + '\n')
    f.flush()


def reset_flag(FLAGS, func, str, v):
    delattr(FLAGS, str)
    func(str, v, '')


def reset_graph():
    del_all_flags(tf.flags.FLAGS)
    tf.reset_default_graph()


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


# Model Hyperparameter tuning
tune_parameter(FLAGS)


# Model Structural Tuning
def exp_attention(FLAGS):
    structure_info = 'attention'
    flags = tf.app.flags

    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_4',
        'Attention')
    FLAGS = tf.app.flags.FLAGS
    tune_structure(FLAGS, structure_info)


def exp_NTN_feature_dim(FLAGS):
    structure_info = 'NTN_feature_dim_8'
    flags = tf.app.flags
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_5',
        'NTN:input_dim=16,feature_map_dim=8,inneract=relu,apply_u=False,'
        'dropout=True,bias=True')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_6',
        'Dense:input_dim=8,output_dim=9,dropout=True,'
        'act=identity,bias=True')
    FLAGS = tf.app.flags.FLAGS
    tune_structure(FLAGS, structure_info)


def exp_identity(FLAGS):
    structure_info = 'act layer all identity'
    flags = tf.app.flags
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_1',
        'GraphConvolution:output_dim=64,act=identity,'
        'dropout=True,bias=True,sparse_inputs=True')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_2',
        'GraphConvolution:input_dim=64,output_dim=32,act=identity,'
        'dropout=True,bias=True,sparse_inputs=False')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_3',
        'GraphConvolution:input_dim=32,output_dim=16,act=identity,'
        'dropout=True,bias=True,sparse_inputs=False')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_4',
        'Average')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_5',
        'NTN:input_dim=16,feature_map_dim=16,inneract=identity,apply_u=False,'
        'dropout=True,bias=True')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_6',
        'Dense:input_dim=16,output_dim=9,dropout=True,'
        'act=identity,bias=True')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_7',
        'Dense:input_dim=9,output_dim=2,dropout=True,'
        'act=identity,bias=True')
    FLAGS = tf.app.flags.FLAGS
    tune_structure(FLAGS, structure_info)


def exp_NTN_bias(FLAGS):
    structure_info = 'NTN no bias'
    flags = tf.app.flags
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_5',
        'NTN:input_dim=8,feature_map_dim=10,inneract=relu,apply_u=False,'
        'dropout=True,bias=False')
    FLAGS = tf.app.flags.FLAGS
    tune_structure(FLAGS, structure_info)


def exp_dot(FLAGS):
    structure_info = 'dot replace NTN'
    flags = tf.app.flags
    reset_flag(FLAGS, flags.DEFINE_integer, 'layer_num', 5)
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_1',
        'GraphConvolution:output_dim=64,act=identity,'
        'dropout=True,bias=True,sparse_inputs=True')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_2',
        'GraphConvolution:input_dim=64,output_dim=32,act=identity,'
        'dropout=True,bias=True,sparse_inputs=False')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_3',
        'GraphConvolution:input_dim=32,output_dim=16,act=identity,'
        'dropout=True,bias=True,sparse_inputs=False')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_4',
        'Average')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_5',
        'dot')
    FLAGS = tf.app.flags.FLAGS
    tune_structure(FLAGS, structure_info)


def exp_2_GCN_layer(FLAGS):
    structure_info = '2 layer GCN'
    flags = tf.app.flags
    reset_flag(FLAGS, flags.DEFINE_integer, 'layer_num', 6)
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_1',
        'GraphConvolution:output_dim=64,act=identity,'
        'dropout=True,bias=True,sparse_inputs=True')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_2',
        'GraphConvolution:input_dim=64,output_dim=32,act=identity,'
        'dropout=True,bias=True,sparse_inputs=False')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_3',
        'Average')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_4',
        'NTN:input_dim=16,feature_map_dim=16,inneract=identity,apply_u=False,'
        'dropout=True,bias=True')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_5',
        'Dense:input_dim=16,output_dim=9,dropout=True,'
        'act=identity,bias=True')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_6',
        'Dense:input_dim=9,output_dim=2,dropout=True,'
        'act=identity,bias=True')
    FLAGS = tf.app.flags.FLAGS
    tune_structure(FLAGS, structure_info)


def exp_change_input_dim(FLAGS):
    structure_info = 'GCN change input dim'
    flags = tf.app.flags
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_1',
        'GraphConvolution:output_dim=32,act=identity,'
        'dropout=True,bias=True,sparse_inputs=True')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_2',
        'GraphConvolution:input_dim=32,output_dim=16,act=identity,'
        'dropout=True,bias=True,sparse_inputs=False')
    FLAGS = tf.app.flags.FLAGS
    tune_structure(FLAGS, structure_info)


def exp_all_bias_false(FLAGS):
    structure_info = 'all bias false'
    flags = tf.app.flags
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_1',
        'GraphConvolution:output_dim=64,act=relu,'
        'dropout=True,bias=False,sparse_inputs=True')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_2',
        'GraphConvolution:input_dim=64,output_dim=32,act=identity,'
        'dropout=True,bias=False,sparse_inputs=False')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_3',
        'GraphConvolution:input_dim=32,output_dim=16,act=identity,'
        'dropout=True,bias=False,sparse_inputs=False')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_4',
        'Average')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_5',
        'NTN:input_dim=16,feature_map_dim=16,inneract=relu,apply_u=False,'
        'dropout=True,bias=False')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_6',
        'Dense:input_dim=16,output_dim=9,dropout=True,'
        'act=relu,bias=False')
    reset_flag(
        FLAGS, flags.DEFINE_string,
        'layer_7',
        'Dense:input_dim=9,output_dim=2,dropout=True,'
        'act=identity,bias=False')
    FLAGS = tf.app.flags.FLAGS
    tune_structure(FLAGS, structure_info)


def exp_iter_20000(FLAGS):
    structure_info = 'iteration equal 20000'
    flags = tf.app.flags
    reset_flag(FLAGS, flags.DEFINE_integer)


def exp_batch_128(FLAGS):
    structure_info = 'batch size equal 128'
    flags = tf.app.flags
    reset_flag(
        FLAGS, flags.DEFINE_integer,
        'batch_size',
        128)


def exp_batch_256(FLAGS):
    structure_info = 'batch size equal 256'
    flags = tf.app.flags
    reset_flag(
        FLAGS, flags.DEFINE_integer,
        'batch_size',
        256)


def exp_batch_512(FLAGS):
    structure_info = 'batch size equal 512'
    flags = tf.app.flags
    reset_flag(
        FLAGS, flags.DEFINE_integer,
        'batch_size',
        512)


def exp_layer_9(FLAGS):
    structure_info = ''
    flags = tf.app.flags


if __name__ == '__main__':
    tune_parameter(FLAGS)
    exp_attention(FLAGS)
