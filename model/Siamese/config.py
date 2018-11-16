import tensorflow as tf

# Hyper-parameters.
flags = tf.app.flags

# For data preprocessing.
""" dataset: aids80nef, aids700nef, linux, imdbmulti. """
dataset = 'aids700nef'
flags.DEFINE_string('dataset', dataset, 'Dataset string.')
if 'aids' in dataset:
    node_feat_name = 'type'
    node_feat_encoder = 'onehot'
    max_nodes = 10
elif dataset == 'linux' or 'imdb' in dataset:
    node_feat_name = None
    node_feat_encoder = 'constant_1'
    if dataset == 'linux':
        max_nodes = 10
    else:
        max_nodes = 90
else:
    assert (False)
flags.DEFINE_integer('max_nodes', max_nodes, 'Maximum number of nodes in a graph.')
flags.DEFINE_string('node_feat_name', node_feat_name, 'Name of the node feature.')
flags.DEFINE_string('node_feat_encoder', node_feat_encoder,
                    'How to encode the node feature.')
""" valid_percentage: (0, 1). """
flags.DEFINE_float('valid_percentage', 0.25,
                   '(# validation graphs) / (# validation + # training graphs.')
""" dist_metric: ged. """
flags.DEFINE_string('dist_metric', 'ged', 'Distance metric to use.')
""" dist_algo: beam80, astar for ged. """
flags.DEFINE_string('dist_algo', 'astar',
                    'Ground-truth distance algorithm to use.')
flags.DEFINE_boolean('bfs_ordering', False, '')
""" coarsening: 'metis_<num_level>', None. """
flags.DEFINE_string('coarsening', None, 'Algorithm for graph coarsening.')

# For model.
""" model: siamese_regression, siamese_ranking, siamese_classification. """
model = 'siamese_regression'
flags.DEFINE_string('model', model, 'Model string.')
flags.DEFINE_integer('batch_size', 2, 'Number of graph pairs in a batch.')
""" dist_norm: True, False. """
flags.DEFINE_boolean('dist_norm', True,
                     'Whether to normalize the distance or not '
                     'when choosing the ground truth distance.')
if model == 'siamese_regression':
    flags.DEFINE_integer('layer_num', 8, 'Number of layers.')
    flags.DEFINE_string(
        'layer_1',
        'GraphConvolution:output_dim=64,dropout=False,bias=True,'
        'act=relu,sparse_inputs=True', '')
    flags.DEFINE_string(
        'layer_2',
        'GraphConvolution:input_dim=64,output_dim=32,dropout=False,bias=True,'
        'act=relu,sparse_inputs=False', '')
    flags.DEFINE_string(
        'layer_3',
        'GraphConvolution:input_dim=32,output_dim=16,dropout=False,bias=True,'
        'act=identity,sparse_inputs=False', '')
    # flags.DEFINE_string(
    #     'layer_3',
    #     'GraphConvolutionAttention:input_dim=32,output_dim=16,dropout=False,bias=True,'
    #     'act=identity,sparse_inputs=False', '')
    flags.DEFINE_string(
        'layer_4',
        'Average', '')
    # flags.DEFINE_string(
    #     'layer_4',
    #     'Attention:input_dim=16,att_times=1,att_num=1,att_weight=True,att_style=dot', '')
    flags.DEFINE_string(
        'layer_5',
        'NTN:input_dim=16,feature_map_dim=16,dropout=False,bias=True,'
        'inneract=relu,apply_u=False', '')
    # flags.DEFINE_string(
    #     'layer_4',
    #     'ANNH:input_dim=16,att_times=1,att_num=1,att_style=dot,att_weight=True,'
    #     'feature_map_dim=16,dropout=False,bias=True,'
    #     'ntn_inneract=relu,apply_u=False,'
    #     'padding_value=0,'
    #     'mne_inneract=sigmoid,mne_method=hist_16,branch_style=anpm', '')
    # flags.DEFINE_string(
    #     'layer_4',
    #     'ANPMD:input_dim=16,att_times=1,att_num=1,att_style=dot,att_weight=True,'
    #     'feature_map_dim=16,dropout=False,bias=True,'
    #     'ntn_inneract=relu,apply_u=False,'
    #     'padding_value=0,'
    #     'mne_inneract=sigmoid,mne_method=hist_16,branch_style=anpm,'
    #     'dense1_dropout=False,dense1_act=relu,dense1_bias=True,dense1_output_dim=8,'
    #     'dense2_dropout=False,dense2_act=relu,dense2_bias=True,dense2_output_dim=4', '')
    # flags.DEFINE_string(
    #     'layer_5', 'Dot:output_dim=1', '')
    # flags.DEFINE_string(
    #     'layer_5',
    #     'Dense:input_dim=80,output_dim=48,dropout=False,bias=True,'
    #     'act=relu', '')
    # flags.DEFINE_string(
    #     'layer_6',
    #     'Dense:input_dim=48,output_dim=32,dropout=False,bias=True,'
    #     'act=relu', '')
    # flags.DEFINE_string(
    #     'layer_5',
    #     'Dense:input_dim=32,output_dim=16,dropout=False,bias=True,'
    #     'act=relu', '')
    flags.DEFINE_string(
        'layer_6',
        'Dense:input_dim=16,output_dim=8,dropout=False,bias=True,'
        'act=relu', '')
    flags.DEFINE_string(
        'layer_7',
        'Dense:input_dim=8,output_dim=4,dropout=False,bias=True,'
        'act=relu', '')
    flags.DEFINE_string(
        'layer_8',
        'Dense:input_dim=4,output_dim=1,dropout=False,bias=True,'
        'act=sigmoid', '')

    # --------------------------------- MNE+CNN ---------------------------------
    # flags.DEFINE_string(
    #     'layer_1',
    #     'GraphConvolution:output_dim=64,dropout=False,bias=True,'
    #     'act=relu,sparse_inputs=True', '')
    # flags.DEFINE_string(
    #     'layer_2',
    #     'GraphConvolution:input_dim=64,output_dim=32,dropout=False,bias=True,'
    #     'act=relu,sparse_inputs=False', '')
    # flags.DEFINE_string(
    #     'layer_3',
    #     'GraphConvolution:input_dim=32,output_dim=16,dropout=False,bias=True,'
    #     'act=identity,sparse_inputs=False', '')
    # # flags.DEFINE_string(
    # #     'layer_1',
    # #     'GraphConvolution:output_dim=64,act=relu,'
    # #     'dropout=True,bias=True,sparse_inputs=True', '')
    # # flags.DEFINE_string(
    # #     'layer_2',
    # #     'GraphConvolution:input_dim=64,output_dim=32,act=relu,'
    # #     'dropout=True,bias=False,sparse_inputs=False', '')
    # # flags.DEFINE_string(
    # #     'layer_3',
    # #     'GraphConvolution:input_dim=32,output_dim=16,act=relu,'
    # #     'dropout=True,bias=False,sparse_inputs=False', '')
    # # flags.DEFINE_string(
    # #     'layer_4',
    # #     'Attention:input_dim=16,att_times=1,att_num=1,att_style=ntn_1,att_weight=True', '')
    # # flags.DEFINE_string(
    # #     'layer_4',
    # #     'Dense:input_dim=16,output_dim=8,dropout=False,'
    # #     'act=relu,bias=True', '')
    # flags.DEFINE_string(
    #     'layer_4',
    #     'Padding:padding_value=0', '')  # Assume the max node # is max_in_dim
    # flags.DEFINE_string(
    #     'layer_5',
    #     'MNE:input_dim=16,dropout=False,inneract=relu', '')
    # # flags.DEFINE_string(
    # #     'layer_6',
    # #     'CNN:start_cnn=True,end_cnn=False,window_size=10,kernel_stride=1,in_channel=1,out_channel=16,'
    # #     'padding=SAME,pool_size=2,dropout=True,act=relu,bias=True', '')
    # # flags.DEFINE_string(
    # #     'layer_7',
    # #     'CNN:start_cnn=False,end_cnn=False,window_size=8,kernel_stride=1,in_channel=16,out_channel=32,'
    # #     'padding=SAME,pool_size=2,dropout=True,act=relu,bias=True', '')
    # # flags.DEFINE_string(
    # #     'layer_8',
    # #     'CNN:start_cnn=False,end_cnn=True,window_size=6,kernel_stride=1,in_channel=32,out_channel=64,'
    # #     'padding=SAME,pool_size=3,dropout=True,act=relu,bias=True', '')
    # flags.DEFINE_string(
    #     'layer_6',
    #     'CNN:start_cnn=True,end_cnn=False,window_size=50,kernel_stride=1,in_channel=1,out_channel=16,'
    #     'padding=SAME,pool_size=3,dropout=True,act=relu,bias=True', '')
    # flags.DEFINE_string(
    #     'layer_7',
    #     'CNN:start_cnn=False,end_cnn=False,window_size=40,kernel_stride=1,in_channel=16,out_channel=32,'
    #     'padding=SAME,pool_size=3,dropout=True,act=relu,bias=True', '')
    # flags.DEFINE_string(
    #     'layer_8',
    #     'CNN:start_cnn=False,end_cnn=False,window_size=30,kernel_stride=1,in_channel=32,out_channel=64,'
    #     'padding=SAME,pool_size=3,dropout=True,act=relu,bias=True', '')
    # flags.DEFINE_string(
    #     'layer_9',
    #     'CNN:start_cnn=False,end_cnn=True,window_size=20,kernel_stride=1,in_channel=64,out_channel=64,'
    #     'padding=SAME,pool_size=4,dropout=True,act=relu,bias=True', '')
    # flags.DEFINE_string(
    #     'layer_10',
    #     'Dense:input_dim=64,output_dim=32,dropout=False,'
    #     'act=identity,bias=True', '')
    # flags.DEFINE_string(
    #     'layer_11',
    #     'Dense:input_dim=32,output_dim=16,dropout=False,'
    #     'act=identity,bias=True', '')
    # flags.DEFINE_string(
    #     'layer_12',
    #     'Dense:input_dim=16,output_dim=1,dropout=False,'
    #     'act=identity,bias=True', '')

    # Start of mse loss.
    """ sim_kernel: gaussian, exp, inverse, identity. """
    sim_kernel = 'exp'
    flags.DEFINE_string('sim_kernel', sim_kernel,
                        'Name of the similarity kernel.')
    if sim_kernel == 'gaussian':
        """ yeta:
         if dist_norm, try 0.6 for nef small, 0.3 for nef, 0.2 for regular;
         else, try 0.01 for nef, 0.001 for regular. """
        flags.DEFINE_float('yeta', 0.01, 'yeta for the gaussian kernel function.')
    elif sim_kernel == 'exp' or sim_kernel == 'inverse':
        flags.DEFINE_float('scale', 1, 'Scale for the exp/inverse kernel function.')

    # End of mse loss.
elif model == 'siamese_ranking':
    flags.DEFINE_integer('layer_num', 8, 'Number of layers.')
    flags.DEFINE_string(
        'layer_1',
        'GraphConvolution:output_dim=32,dropout=False,bias=True,'
        'act=relu,sparse_inputs=True', '')
    flags.DEFINE_string(
        'layer_2',
        'GraphConvolution:input_dim=32,output_dim=16,dropout=False,bias=True,'
        'act=relu,sparse_inputs=False', '')
    flags.DEFINE_string(
        'layer_3',
        'GraphConvolution:input_dim=16,output_dim=8,dropout=False,bias=True,'
        'act=identity,sparse_inputs=False', '')
    # flags.DEFINE_string(
    #     'layer_4',
    #     'Average', '')
    # flags.DEFINE_string(
    #     'layer_4',
    #     'ANPM:input_dim=8,att_times=1,att_num=1,att_style=ntn_1,att_weight=True,'
    #     'feature_map_dim=16,dropout=False,bias=True,'
    #     'ntn_inneract=relu,apply_u=False,'
    #     'padding_value=0,'
    #     'mne_inneract=sigmoid,num_bins=16', '')
    # flags.DEFINE_string(
    #     'layer_4',
    #     'ANPM:input_dim=8,att_times=1,att_num=1,att_style=ntn_1,att_weight=True,'
    #     'feature_map_dim=16,dropout=False,bias=True,'
    #     'ntn_inneract=relu,apply_u=False,'
    #     'padding_value=0,'
    #     'mne_inneract=sigmoid,method=arg_max_naive,norm=True', '')
    flags.DEFINE_string(
        'layer_4',
        'Attention:input_dim=8,att_times=1,att_num=1,att_style=dot,att_weight=True,', '')
    # flags.DEFINE_string(
    #     'layer_5', 'Dot:output_dim=2', '')
    flags.DEFINE_string(
        'layer_5',
        'NTN:input_dim=8,feature_map_dim=8,inneract=relu,apply_u=False,'
        'dropout=False,bias=True', '')
    flags.DEFINE_string(
        'layer_6',
        'Dense:input_dim=8,output_dim=4,dropout=False,bias=True,'
        'act=relu', '')
    flags.DEFINE_string(
        'layer_7',
        'Dense:input_dim=4,output_dim=2,dropout=False,bias=True,'
        'act=relu', '')  # identity
    flags.DEFINE_string(
        'layer_8',
        'Dense:input_dim=2,output_dim=1,dropout=False,bias=True,'
        'act=identity', '')  # identity
    # # Start of hinge loss.
    """ delta and gamma: depend on whether dist_norm is True of False. """
    flags.DEFINE_float('delta', 0.,
                       'Margin between positive pairs ground truth scores'
                       'and negative pairs scores  ground truth scores')
    flags.DEFINE_float('gamma', 0.6,
                       'Margin between positive pairs prediction scores'
                       'and negative pairs prediction scores')
    flags.DEFINE_integer('num_neg', 8, 'Number of negative samples.')
    flags.DEFINE_integer('top_k', 0, 'sample positive & negative pairs from top n samples after sort')
    # flags.DEFINE_float('pos_thresh', 0.35, 'sample positive & negative pairs from top n samples after sort')

    # End of hinge loss.
elif model == 'siamese_classification':
    flags.DEFINE_integer('layer_num', 8, 'Number of layers.')
    flags.DEFINE_string(
        'layer_1',
        'GraphConvolution:output_dim=64,dropout=False,bias=True,'
        'act=relu,sparse_inputs=True', '')
    # flags.DEFINE_string(
    #     'layer_2',
    #     'Coarsening:pool_style=avg', '')
    flags.DEFINE_string(
        'layer_2',
        'GraphConvolution:input_dim=64,output_dim=32,dropout=False,bias=True,'
        'act=relu,sparse_inputs=False', '')
    # flags.DEFINE_string(
    #     'layer_4',
    #     'Coarsening:pool_style=avg', '')
    flags.DEFINE_string(
        'layer_3',
        'GraphConvolution:input_dim=32,output_dim=16,dropout=False,bias=True,'
        'act=identity,sparse_inputs=False', '')
    # flags.DEFINE_string(
    #     'layer_6',
    #     'Coarsening:pool_style=avg', '')
    # flags.DEFINE_string(
    #     'layer_4',
    #     'Average', '')
    # flags.DEFINE_string(
    #     'layer_4',
    #     'Attention:input_dim=16,att_times=1,att_num=1,att_style=dot,att_weight=True', '')
    # flags.DEFINE_string(
    #     'layer_8', 'Dot:output_dim=1', '')
    # flags.DEFINE_string(
    #     'layer_8',
    #     'SLM:input_dim=16,output_dim=16,act=tanh,dropout=False,bias=True', '')
    # flags.DEFINE_string(
    #     'layer_5',
    #     'NTN:input_dim=16,feature_map_dim=16,dropout=False,bias=True,'
    #     'inneract=relu,apply_u=False', '')
    flags.DEFINE_string(
        'layer_4',
        'ANPM:input_dim=16,att_times=1,att_num=1,att_style=dot,att_weight=True,'
        'feature_map_dim=16,dropout=False,bias=True,'
        'ntn_inneract=relu,apply_u=False,'
        'padding_value=0,'
        'mne_inneract=sigmoid,method=hist_16,norm=True', '')
    flags.DEFINE_string(
        'layer_5',
        'Dense:input_dim=32,output_dim=16,dropout=False,bias=True,'
        'act=relu', '')
    flags.DEFINE_string(
        'layer_6',
        'Dense:input_dim=16,output_dim=8,dropout=False,bias=True,'
        'act=relu', '')
    flags.DEFINE_string(
        'layer_7',
        'Dense:input_dim=8,output_dim=4,dropout=False,bias=True,'
        'act=relu', '')
    flags.DEFINE_string(
        'layer_8',
        'Dense:input_dim=4,output_dim=2,dropout=False,bias=True,'
        'act=identity', '')

    # Start of cross entropy loss.
    """
    aids700nef:    0.65 0.74 0.83 0.89 0.95 1.0  1.2  1.25 1.49
    linux:         0.25 0.35 0.43 0.53 0.58 0.67 0.78 0.89 1.1
    imdb1kcoarse:  0.45 0.6  0.77 0.88 0.99 1.15 1.35 1.65 2.1
    """
    if 'aids' in dataset:
        thresh = 0.95
        # thresh = 0.65
    elif dataset == 'linux':
        thresh = 0.58
        # thresh = 0.25
    elif 'imdb' in dataset:
        thresh = 0.99
        # thresh = 0.45
    else:
        assert (False)
    assert (flags.FLAGS.dist_norm)
    flags.DEFINE_float('thresh_train_pos', thresh,
                       'Threshold below which train pairs are similar.')
    flags.DEFINE_float('thresh_train_neg', thresh,
                       'Threshold above which train pairs are dissimilar.')
    flags.DEFINE_float('thresh_val_test_pos', thresh,
                       'Threshold that binarizes test pairs.')
    flags.DEFINE_float('thresh_val_test_neg', thresh,
                       'Threshold that binarizes test pairs.')
    # End of cross entropy loss.

# Start of graph loss.
""" graph_loss: '1st', None. """
graph_loss = None
flags.DEFINE_string('graph_loss', graph_loss, 'Loss function(s) to use.')
if graph_loss:
    flags.DEFINE_float('graph_loss_alpha', 0.,
                       'Weight parameter for the graph loss function.')

# Generater, Repeater and Permutater
flags.DEFINE_string('fake_generation', None,
                    'Whether to generate fake graphs for all graphs or not. '
                    '1st represents top num, and 2nd represents fake num.')
flags.DEFINE_string('top_repeater', None,
                    'Whether to generate fake graphs for top k sim '
                    'graphs or not. 1st represents top num, and 2nd represents repeat num.')
flags.DEFINE_boolean('random_permute', False,
                     'Whether to random permute nodes of graphs in training or not.')
# End of graph loss.

flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0,
                   'Weight for L2 loss on embedding matrix.')
""" learning_rate: 0.01 recommended. """
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')

# For training and validating.
flags.DEFINE_integer('gpu', -1, 'Which gpu to use.')  # -1: cpu
flags.DEFINE_integer('iters', 2, 'Number of iterations to train.')
flags.DEFINE_integer('iters_val_start', 2,
                     'Number of iterations to start validation.')
flags.DEFINE_integer('iters_val_every', 2, 'Frequency of validation.')

# For testing.
flags.DEFINE_boolean('plot_results', True,
                     'Whether to plot the results '
                     '(involving all baselines) or not.')

FLAGS = tf.app.flags.FLAGS

# import tensorflow as tf
#
# # Hyper-parameters.
# flags = tf.app.flags
#
# # For data preprocessing.
# """ dataset: aids80nef, aids700nef, linux, imdbmulti. """
# dataset = 'imdbmulti'
# flags.DEFINE_string('dataset', dataset, 'Dataset string.')
# if 'aids' in dataset:
#     node_feat_name = 'type'
#     node_feat_encoder = 'onehot'
#     max_nodes = 10
# elif dataset == 'linux' or 'imdb' in dataset:
#     node_feat_name = None
#     node_feat_encoder = 'constant_1'
#     if dataset == 'linux':
#         max_nodes = 10
#     else:
#         max_nodes = 90
# else:
#     assert (False)
# flags.DEFINE_integer('max_nodes', max_nodes, 'Maximum number of nodes in a graph.')
# flags.DEFINE_string('node_feat_name', node_feat_name, 'Name of the node feature.')
# flags.DEFINE_string('node_feat_encoder', node_feat_encoder,
#                     'How to encode the node feature.')
# """ valid_percentage: (0, 1). """
# flags.DEFINE_float('valid_percentage', 0.25,
#                    '(# validation graphs) / (# validation + # training graphs.')
# """ dist_metric: ged. """
# flags.DEFINE_string('dist_metric', 'ged', 'Distance metric to use.')
# """ dist_algo: beam80, astar for ged. """
# flags.DEFINE_string('dist_algo', 'astar',
#                     'Ground-truth distance algorithm to use.')
# flags.DEFINE_boolean('bfs_ordering', False, '')
# """ coarsening: 'metis_<num_level>', None. """
# flags.DEFINE_string('coarsening', None, 'Algorithm for graph coarsening.')
#
# # For model.
# """ model: siamese_regression, siamese_ranking, siamese_classification. """
# model = 'siamese_regression'
# flags.DEFINE_string('model', model, 'Model string.')
# flags.DEFINE_integer('batch_size', 256, 'Number of graph pairs in a batch.')
# """ dist_norm: True, False. """
# flags.DEFINE_boolean('dist_norm', True,
#                      'Whether to normalize the distance or not '
#                      'when choosing the ground truth distance.')
# if model == 'siamese_regression':
#     flags.DEFINE_integer('layer_num', 6, 'Number of layers.')
#     flags.DEFINE_string(
#         'layer_1',
#         'GraphConvolution:output_dim=64,dropout=False,bias=True,'
#         'act=relu,sparse_inputs=True', '')
#     # flags.DEFINE_string(
#     #     'layer_2',
#     #     'Coarsening:pool_style=max', '')
#     flags.DEFINE_string(
#         'layer_2',
#         'GraphConvolution:input_dim=64,output_dim=32,dropout=False,bias=True,'
#         'act=relu,sparse_inputs=False', '')
#     # flags.DEFINE_string(
#     #     'layer_4',
#     #     'Coarsening:pool_style=max', '')
#     flags.DEFINE_string(
#         'layer_3',
#         'GraphConvolution:input_dim=32,output_dim=16,dropout=False,bias=True,'
#         'act=identity,sparse_inputs=False', '')
#     # flags.DEFINE_string(
#     #     'layer_6',
#     #     'Coarsening:pool_style=max', '')
#     # flags.DEFINE_string(
#     #     'layer_3',
#     #     'GraphConvolutionAttention:input_dim=32,output_dim=16,dropout=False,bias=True,'
#     #     'act=identity,sparse_inputs=False', '')
#     # flags.DEFINE_string(
#     #     'layer_7',
#     #     'Average', '')
#     # flags.DEFINE_string(
#     #     'layer_4',
#     #     'Attention:input_dim=16,att_times=1,att_num=1,att_weight=False,att_style=dot', '')
#     # flags.DEFINE_string(
#     #     'layer_5',
#     #     'Dot:output_dim=1', '')
#     # flags.DEFINE_string(
#     #     'layer_8',
#     #     'NTN:input_dim=16,feature_map_dim=16,dropout=False,bias=True,'
#     #     'inneract=relu,apply_u=False', '')
#     # flags.DEFINE_string(
#     #     'layer_4',
#     #     'ANPM:input_dim=32,att_times=1,att_num=1,att_style=dot,att_weight=True,'
#     #     'feature_map_dim=32,dropout=False,bias=True,'
#     #     'ntn_inneract=relu,apply_u=False,'
#     #     'padding_value=0,'
#     #     'mne_inneract=sigmoid_0.5,mne_method=hist_32,branch_style=anpm', '')
#     flags.DEFINE_string(
#         'layer_4',
#         'ANPMD:input_dim=16,att_times=1,att_num=1,att_style=dot,att_weight=True,'
#         'feature_map_dim=16,dropout=False,bias=True,'
#         'ntn_inneract=relu,apply_u=False,'
#         'padding_value=0,'
#         'mne_inneract=sigmoid_0.5,mne_method=hist_16,branch_style=anpm,'
#         'dense1_dropout=False,dense1_act=relu,dense1_bias=True,dense1_output_dim=8,'
#         'dense2_dropout=False,dense2_act=relu,dense2_bias=True,dense2_output_dim=4', '')
#     # flags.DEFINE_string(
#     #     'layer_5', 'Dot:output_dim=1', '')
#     # flags.DEFINE_string(
#     #     'layer_5',
#     #     'Dense:input_dim=64,output_dim=32,dropout=False,bias=True,'
#     #     'act=relu', '')
#     # flags.DEFINE_string(
#     #     'layer_6',
#     #     'Dense:input_dim=32,output_dim=16,dropout=False,bias=True,'
#     #     'act=relu', '')
#     # flags.DEFINE_string(
#     #     'layer_9',
#     #     'Dense:input_dim=16,output_dim=8,dropout=False,bias=True,'
#     #     'act=relu', '')
#     flags.DEFINE_string(
#         'layer_5',
#         'Dense:input_dim=8,output_dim=4,dropout=False,bias=True,'
#         'act=relu', '')
#     flags.DEFINE_string(
#         'layer_6',
#         'Dense:input_dim=4,output_dim=1,dropout=False,bias=True,'
#         'act=sigmoid', '')
#
#     # Start of mse loss.
#     """ sim_kernel: gaussian, exp, inverse, identity. """
#     sim_kernel = 'inverse'
#     flags.DEFINE_string('sim_kernel', sim_kernel,
#                         'Name of the similarity kernel.')
#     if sim_kernel == 'gaussian':
#         """ yeta:
#          if dist_norm, try 0.6 for nef small, 0.3 for nef, 0.2 for regular;
#          else, try 0.01 for nef, 0.001 for regular. """
#         flags.DEFINE_float('yeta', 0.3, 'yeta for the gaussian kernel function.')
#     elif sim_kernel == 'exp' or sim_kernel == 'inverse':
#         flags.DEFINE_float('scale', 0.6, 'Scale for the exp/inverse kernel function.')
#     # End of mse loss.
# elif model == 'siamese_ranking':
#     flags.DEFINE_integer('layer_num', 8, 'Number of layers.')
#     flags.DEFINE_string(
#         'layer_1',
#         'GraphConvolution:output_dim=32,dropout=False,bias=True,'
#         'act=relu,sparse_inputs=True', '')
#     flags.DEFINE_string(
#         'layer_2',
#         'GraphConvolution:input_dim=32,output_dim=16,dropout=False,bias=True,'
#         'act=relu,sparse_inputs=False', '')
#     flags.DEFINE_string(
#         'layer_3',
#         'GraphConvolution:input_dim=16,output_dim=8,dropout=False,bias=True,'
#         'act=identity,sparse_inputs=False', '')
#     # flags.DEFINE_string(
#     #     'layer_4',
#     #     'Average', '')
#     # flags.DEFINE_string(
#     #     'layer_4',
#     #     'ANPM:input_dim=8,att_times=1,att_num=1,att_style=ntn_1,att_weight=True,'
#     #     'feature_map_dim=16,dropout=False,bias=True,'
#     #     'ntn_inneract=relu,apply_u=False,'
#     #     'padding_value=0,'
#     #     'mne_inneract=sigmoid,num_bins=16', '')
#     # flags.DEFINE_string(
#     #     'layer_4',
#     #     'ANPM:input_dim=8,att_times=1,att_num=1,att_style=ntn_1,att_weight=True,'
#     #     'feature_map_dim=16,dropout=False,bias=True,'
#     #     'ntn_inneract=relu,apply_u=False,'
#     #     'padding_value=0,'
#     #     'mne_inneract=sigmoid,method=arg_max_naive,norm=True', '')
#     flags.DEFINE_string(
#         'layer_4',
#         'Attention:input_dim=8,att_times=1,att_num=1,att_style=dot,att_weight=True,', '')
#     # flags.DEFINE_string(
#     #     'layer_5', 'Dot:output_dim=2', '')
#     flags.DEFINE_string(
#         'layer_5',
#         'NTN:input_dim=8,feature_map_dim=8,inneract=relu,apply_u=False,'
#         'dropout=False,bias=True', '')
#     flags.DEFINE_string(
#         'layer_6',
#         'Dense:input_dim=8,output_dim=4,dropout=False,bias=True,'
#         'act=relu', '')
#     flags.DEFINE_string(
#         'layer_7',
#         'Dense:input_dim=4,output_dim=2,dropout=False,bias=True,'
#         'act=relu', '')  # identity
#     flags.DEFINE_string(
#         'layer_8',
#         'Dense:input_dim=2,output_dim=1,dropout=False,bias=True,'
#         'act=identity', '')  # identity
#     # # Start of hinge loss.
#     """ delta and gamma: depend on whether dist_norm is True of False. """
#     flags.DEFINE_float('delta', 0.,
#                        'Margin between positive pairs ground truth scores'
#                        'and negative pairs scores  ground truth scores')
#     flags.DEFINE_float('gamma', 0.6,
#                        'Margin between positive pairs prediction scores'
#                        'and negative pairs prediction scores')
#     flags.DEFINE_integer('num_neg', 8, 'Number of negative samples.')
#     flags.DEFINE_integer('top_k', 0, 'sample positive & negative pairs from top n samples after sort')
#     # flags.DEFINE_float('pos_thresh', 0.35, 'sample positive & negative pairs from top n samples after sort')
#
#     # End of hinge loss.
# elif model == 'siamese_classification':
#     flags.DEFINE_integer('layer_num', 8, 'Number of layers.')
#     flags.DEFINE_string(
#         'layer_1',
#         'GraphConvolution:output_dim=64,dropout=False,bias=True,'
#         'act=relu,sparse_inputs=True', '')
#     # flags.DEFINE_string(
#     #     'layer_2',
#     #     'Coarsening:pool_style=avg', '')
#     flags.DEFINE_string(
#         'layer_2',
#         'GraphConvolution:input_dim=64,output_dim=32,dropout=False,bias=True,'
#         'act=relu,sparse_inputs=False', '')
#     # flags.DEFINE_string(
#     #     'layer_4',
#     #     'Coarsening:pool_style=avg', '')
#     flags.DEFINE_string(
#         'layer_3',
#         'GraphConvolution:input_dim=32,output_dim=16,dropout=False,bias=True,'
#         'act=identity,sparse_inputs=False', '')
#     # flags.DEFINE_string(
#     #     'layer_6',
#     #     'Coarsening:pool_style=avg', '')
#     # flags.DEFINE_string(
#     #     'layer_4',
#     #     'Average', '')
#     # flags.DEFINE_string(
#     #     'layer_4',
#     #     'Attention:input_dim=16,att_times=1,att_num=1,att_style=dot,att_weight=True', '')
#     # flags.DEFINE_string(
#     #     'layer_8', 'Dot:output_dim=1', '')
#     # flags.DEFINE_string(
#     #     'layer_8',
#     #     'SLM:input_dim=16,output_dim=16,act=tanh,dropout=False,bias=True', '')
#     # flags.DEFINE_string(
#     #     'layer_5',
#     #     'NTN:input_dim=16,feature_map_dim=16,dropout=False,bias=True,'
#     #     'inneract=relu,apply_u=False', '')
#     # flags.DEFINE_string(
#     #     'layer_4',
#     #     'ANPM:input_dim=16,att_times=1,att_num=1,att_style=dot,att_weight=True,'
#     #     'feature_map_dim=16,dropout=False,bias=True,'
#     #     'ntn_inneract=relu,apply_u=False,'
#     #     'padding_value=0,'
#     #     'mne_inneract=sigmoid_0.5,method=hist_16,norm=True', '')
#     flags.DEFINE_string(
#         'layer_4',
#         'ANPMD:input_dim=16,att_times=1,att_num=1,att_style=dot,att_weight=True,'
#         'feature_map_dim=16,dropout=False,bias=True,'
#         'ntn_inneract=relu,apply_u=False,'
#         'padding_value=0,'
#         'mne_inneract=sigmoid_0.5,method=hist_16,branch_style=anpm,'
#         'dense1_dropout=False,dense1_act=relu,dense1_bias=True,dense1_output_dim=8,'
#         'dense2_dropout=False,dense2_act=relu,dense2_bias=True,dense2_output_dim=4', '')
#     # flags.DEFINE_string(
#     #     'layer_5',
#     #     'Dense:input_dim=32,output_dim=16,dropout=False,bias=True,'
#     #     'act=relu', '')
#     # flags.DEFINE_string(
#     #     'layer_6',
#     #     'Dense:input_dim=16,output_dim=8,dropout=False,bias=True,'
#     #     'act=relu', '')
#     flags.DEFINE_string(
#         'layer_5',
#         'Dense:input_dim=8,output_dim=4,dropout=False,bias=True,'
#         'act=relu', '')
#     flags.DEFINE_string(
#         'layer_6',
#         'Dense:input_dim=4,output_dim=1,dropout=False,bias=True,'
#         'act=identity', '')
#
#     # MNE+CNN
#     # flags.DEFINE_string(
#     #     'layer_1',
#     #     'GraphConvolution:output_dim=64,act=relu,'
#     #     'dropout=True,bias=True,sparse_inputs=True', '')
#     # flags.DEFINE_string(
#     #     'layer_2',
#     #     'GraphConvolution:input_dim=64,output_dim=32,act=relu,'
#     #     'dropout=True,bias=False,sparse_inputs=False', '')
#     # flags.DEFINE_string(
#     #     'layer_3',
#     #     'GraphConvolution:input_dim=32,output_dim=16,act=relu,'
#     #     'dropout=True,bias=False,sparse_inputs=False', '')
#     # flags.DEFINE_string(
#     #     'layer_4',
#     #     'Attention:input_dim=16,att_times=1,att_num=1,att_style=ntn_1,att_weight=True', '')
#     # # flags.DEFINE_string(
#     # #     'layer_4',
#     # #     'Dense:input_dim=16,output_dim=8,dropout=False,'
#     # #     'act=relu,bias=True', '')
#     # # flags.DEFINE_string(
#     # #     'layer_5',
#     # #     'Padding:max_in_dims=10,padding_value=0', '')  # Assume the max node # is max_in_dim
#     # flags.DEFINE_string(
#     #     'layer_5',
#     #     'MNE:input_dim=16,dropout=False,inneract=relu', '')
#     # flags.DEFINE_string(
#     #     'layer_6',
#     #     'CNN:start_cnn=True,end_cnn=False,window_size=9,kernel_stride=1,in_channel=1,out_channel=32,'
#     #     'padding=SAME,pool_size=4,dropout=True,act=relu,bias=True', '')
#     # flags.DEFINE_string(
#     #     'layer_7',
#     #     'CNN:start_cnn=False,end_cnn=True,window_size=8,kernel_stride=1,in_channel=32,out_channel=64,'
#     #     'padding=SAME,pool_size=4,dropout=True,act=relu,bias=True', '')
#     # flags.DEFINE_string(
#     #     'layer_8',
#     #     'Dense:input_dim=64,output_dim=2,dropout=False,'
#     #     'act=identity,bias=True', '')
#
#     # Start of cross entropy loss.
#     """
#     aids700nef:    0.65 0.74 0.83 0.89 0.95 1.0  1.2  1.25 1.49
#     linux:         0.25 0.35 0.43 0.53 0.58 0.67 0.78 0.89 1.1
#     imdb1kcoarse:  0.45 0.6  0.77 0.88 0.99 1.15 1.35 1.65 2.1
#     """
#     if 'aids' in dataset:
#         thresh = 0.95
#         # thresh = 0.65
#     elif dataset == 'linux':
#         thresh = 0.58
#         # thresh = 0.25
#     elif 'imdb' in dataset:
#         thresh = 0.99
#         # thresh = 0.45
#     else:
#         assert (False)
#     assert (flags.FLAGS.dist_norm)
#     flags.DEFINE_float('thresh_train_pos', thresh,
#                        'Threshold below which train pairs are similar.')
#     flags.DEFINE_float('thresh_train_neg', thresh,
#                        'Threshold above which train pairs are dissimilar.')
#     flags.DEFINE_float('thresh_val_test_pos', thresh,
#                        'Threshold that binarizes test pairs.')
#     flags.DEFINE_float('thresh_val_test_neg', thresh,
#                        'Threshold that binarizes test pairs.')
#     # End of cross entropy loss.
#
# # Start of graph loss.
# """ graph_loss: '1st', None. """
# graph_loss = None
# flags.DEFINE_string('graph_loss', graph_loss, 'Loss function(s) to use.')
# if graph_loss:
#     flags.DEFINE_float('graph_loss_alpha', 0.,
#                        'Weight parameter for the graph loss function.')
# flags.DEFINE_string('fake_generation', None,
#                     'Whether to generate fake graphs for all '
#                     'graphs or not.')  # 'fake_10'
# flags.DEFINE_string('top_repeater', None,
#                     'Whether to generate fake graphs for top 20 sim '
#                     'graphs or not.')  # '20_repeat_10'
# flags.DEFINE_boolean('random_permute', False,
#                      'Whether to random permute nodes of graphs in training or not.')
# # End of graph loss.
#
# flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')
# flags.DEFINE_float('weight_decay', 0,
#                    'Weight for L2 loss on embedding matrix.')
# """ learning_rate: 0.01 recommended. """
# flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
#
# # For training and validating.
# flags.DEFINE_integer('gpu', -1, 'Which gpu to use.')  # -1: cpu
# flags.DEFINE_integer('iters', 8000, 'Number of iterations to train.')
# flags.DEFINE_integer('iters_val_start', 7000,
#                      'Number of iterations to start validation.')
# flags.DEFINE_integer('iters_val_every', 50, 'Frequency of validation.')
#
# # For testing.
# flags.DEFINE_boolean('plot_results', True,
#                      'Whether to plot the results '
#                      '(involving all baselines) or not.')
#
# # flags.DEFINE_string('special', 'Temporarily turn on normalization for concat anpm features',
# #                      'Special thing to say.')
#
# FLAGS = tf.app.flags.FLAGS
