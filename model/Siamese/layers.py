from config import FLAGS
from utils_siamese import dot
from inits import *
import tensorflow as tf


class Layer(object):
    """ Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = get_layer_name(self)
        self.name = name
        self.vars = {}
        self.sparse_inputs = False

    def get_name(self):
        return self.name

    def produce_graph_level_emb(self):
        return False

    def produce_node_atts(self):
        return False

    def merge_graph_level_embs(self):
        return False

    def __call__(self, inputs):
        return self._call(inputs)

    def _call(self, inputs):
        raise NotImplementedError()

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '_weights/' + var, self.vars[var])

    def handle_dropout(self, dropout_bool):
        if dropout_bool:
            self.dropout = FLAGS.dropout
        else:
            self.dropout = 0.


class Dense(Layer):
    """ Dense layer. """

    def __init__(self, input_dim, output_dim, dropout, act, bias, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.bias = bias
        self.act = act

        self.handle_dropout(dropout)

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        self._log_vars()

    def _call(self, inputs):
        if type(inputs) is list:
            rtn = []
            for input in inputs:
                rtn.append(self._call_one_mat(input))
            return rtn
        else:
            return self._call_one_mat(inputs)

    def _call_one_mat(self, inputs):
        x = inputs

        # dropout
        x = tf.nn.dropout(x, 1 - self.dropout)

        # transform
        output = dot(x, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """ Graph convolution layer. """

    def __init__(self, input_dim, output_dim, dropout,
                 sparse_inputs, act, bias,
                 featureless, num_supports, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.support = None
        self.act = act

        self.handle_dropout(dropout)

        with tf.variable_scope(self.name + '_vars'):
            for i in range(num_supports):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        self._log_vars()

    def _call(self, inputs):
        assert (type(inputs) is list and inputs)
        if type(inputs[0]) is list:
            # Double list.
            rtn = []
            for input in inputs:
                assert (len(input) == 3)
                rtn.append(self._call_one_graph(input))
            return rtn
        else:
            assert (len(inputs) == 3)
            return self._call_one_graph(inputs)

    def _call_one_graph(self, input):
        x = input[0]
        self.laplacians = input[1]
        num_features_nonzero = input[2]

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        support_list = []
        for i in range(len(self.laplacians)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.laplacians[i], pre_sup, sparse=True)
            support_list.append(support)
        output = tf.add_n(support_list)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolutionAttention(GraphConvolution):
    """ Graph convolution with attention layer. """

    def __init__(self, input_dim, output_dim, dropout, sparse_inputs, act,
                 bias, featureless, num_supports, **kwargs):
        super(GraphConvolutionAttention, self).__init__(input_dim, output_dim,
                                                        dropout, sparse_inputs, act, bias, featureless, num_supports,
                                                        **kwargs)

    def _call_one_graph(self, input):
        x = super(GraphConvolutionAttention, self)._call_one_graph(input)
        L = tf.sparse_tensor_dense_matmul(self.laplacians[0], tf.eye(tf.shape(x)[0]))
        degree_att = -tf.log(tf.reshape(tf.diag_part(L), [-1, 1]))
        output = tf.multiply(x, degree_att)
        return output


class Coarsening(Layer):
    """Coarsening layer. """

    def __init__(self, pool_style, **kwargs):
        super(Coarsening, self).__init__(**kwargs)
        self.pool_style = pool_style

    def _call(self, inputs):
        if type(inputs) is list:
            rtn = []
            for input in inputs:
                rtn.append(self._call_one_graph(input))
            return rtn
        else:
            return self._call_one_graph(inputs)

    def _call_one_graph(self, input):
        if self.pool_style == 'max':
            return self._max_pool(input, 2)  # always pool by 2
        elif self.pool_style == 'avg':
            return self._avg_pool(input, 2)  # always pool by 2
        else:
            raise RuntimeError('Unknown pooling style {}'.format(
                self.pool_style))

    def _max_pool(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 0)  # 1 x M x F
            x = tf.expand_dims(x, 3)  # 1 x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1, p, 1, 1], strides=[1, p, 1, 1], padding='SAME')
            # tf.maximum
            x = tf.squeeze(x, [0, 3])  # M/p x F
            return x
        else:
            return x

    def _avg_pool(self, x, p):
        """Average pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 0)  # 1 x M x F
            x = tf.expand_dims(x, 3)  # 1 x M x F x 1
            x = tf.nn.avg_pool(x, ksize=[1, p, 1, 1], strides=[1, p, 1, 1], padding='SAME')
            x = tf.squeeze(x, [0, 3])  # M/p x F
            return x
        else:
            return x


""" ### Start of generating node embeddings into graoh-level embeddings. ### """


class Average(Layer):
    """ Average layer. """

    def __init__(self, **kwargs):
        super(Average, self).__init__(**kwargs)

    def produce_graph_level_emb(self):
        return True

    def merge_graph_level_embs(self):
        return False

    def _call(self, inputs):
        if type(inputs) is list:
            rtn = []
            for input in inputs:
                rtn.append(self._call_one_mat(input))
            return rtn
        else:
            return self._call_one_mat(inputs)

    def _call_one_mat(self, input):
        x = input  # (N, D)
        output = tf.reshape(tf.reduce_mean(x, 0), [1, -1])  # (1, D)
        return output


class Attention(Average):
    """ Attention layer."""

    def __init__(self, input_dim, att_times, att_num, att_style, att_weight, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.emb_dim = input_dim  # same dimension D as input embeddings
        self.att_times = att_times
        self.att_num = att_num
        self.att_style = att_style
        self.att_weight = att_weight
        assert (self.att_times >= 1)
        assert (self.att_num >= 1)
        assert (self.att_style == 'dot' or self.att_style == 'slm' or
                'ntn_' in self.att_style)

        with tf.variable_scope(self.name + '_vars'):
            for i in range(self.att_num):
                self.vars['W_' + str(i)] = \
                    glorot([self.emb_dim, self.emb_dim],
                           name='W_' + str(i))
                if self.att_style == 'slm':
                    self.interact_dim = 1
                    self.vars['NTN_V_' + str(i)] = \
                        glorot([self.interact_dim, 2 * self.emb_dim],
                               name='NTN_V_' + str(i))
                if 'ntn_' in self.att_style:
                    self.interact_dim = int(self.att_style[4])
                    self.vars['NTN_V_' + str(i)] = \
                        glorot([self.interact_dim, 2 * self.emb_dim],
                               name='NTN_V_' + str(i))
                    self.vars['NTN_W_' + str(i)] = \
                        glorot([self.interact_dim, self.emb_dim, self.emb_dim],
                               name='NTN_W_' + str(i))
                    self.vars['NTN_U_' + str(i)] = \
                        glorot([self.interact_dim, 1],
                               name='NTN_U_' + str(i))
                    self.vars['NTN_b_' + str(i)] = \
                        zeros([self.interact_dim],
                              name='NTN_b_' + str(i))

        self._log_vars()

    def produce_node_atts(self):
        return True

    def _call_one_mat(self, inputs):
        outputs = []
        for i in range(self.att_num):
            acts = [inputs]
            assert (self.att_times >= 1)
            output = None
            for _ in range(self.att_times):
                x = acts[-1]  # x is N*D
                temp = tf.reshape(tf.reduce_mean(x, 0), [1, -1])  # (1, D)
                h_avg = tf.tanh(tf.matmul(temp, self.vars['W_' + str(i)])) if \
                    self.att_weight else temp
                self.att = self._gen_att(x, h_avg, i)
                output = tf.matmul(tf.reshape(self.att, [1, -1]), x)  # (1, D)
                x_new = tf.multiply(x, self.att)
                acts.append(x_new)
            outputs.append(output)
        return tf.concat(outputs, 1)

    def _gen_att(self, x, h_avg, i):
        if self.att_style == 'dot':
            return interact_two_sets_of_vectors(
                x, h_avg, 1,  # interact only once
                W=[tf.eye(self.emb_dim)],
                act=tf.sigmoid)
        elif self.att_style == 'slm':
            # return tf.sigmoid(tf.matmul(concat, self.vars['a_' + str(i)]))
            return interact_two_sets_of_vectors(
                x, h_avg, self.interact_dim,
                V=self.vars['NTN_V_' + str(i)],
                act=tf.sigmoid)
        else:
            assert ('ntn_' in self.att_style)
            return interact_two_sets_of_vectors(
                x, h_avg, self.interact_dim,
                V=self.vars['NTN_V_' + str(i)],
                W=self.vars['NTN_W_' + str(i)],
                b=self.vars['NTN_b_' + str(i)],
                act=tf.sigmoid,
                U=self.vars['NTN_U_' + str(i)])


""" ### End of generating node embeddings into graoh-level embeddings. ### """


class Padding(Layer):
    """ Padding layer. """

    def __init__(self, padding_value, **kwargs):
        super(Padding, self).__init__(**kwargs)
        self.padding_value = padding_value
        self.max_in_dims = FLAGS.max_nodes

    def _call(self, inputs):
        if type(inputs) is list:
            rtn = []
            for input in inputs:
                rtn.append(self._call_one_mat(input))
            return rtn
        else:
            return self._call_one_mat(inputs)

    def _call_one_mat(self, inputs):
        s = tf.shape(inputs)
        # paddings = [[0, m - s[i]] for (i, m) in enumerate(self.max_in_dims)]
        paddings = [[0, self.max_in_dims - s[0]], [0, 0]]  # Assume inputs dim is N*D
        outs = tf.pad(inputs, paddings, 'CONSTANT', constant_values=self.padding_value)
        return tf.reshape(outs, [self.max_in_dims, -1])  # TODO: ?????


""" ############# Start of merging two graph-level embeddings. ############# """


class Merge(Layer):
    """ Merge layer. """

    def __init__(self, **kwargs):
        super(Merge, self).__init__(**kwargs)

    def produce_graph_level_emb(self):
        return False

    def merge_graph_level_embs(self):
        return True

    def _call(self, inputs):
        assert (type(inputs) is list and inputs)
        if type(inputs[0]) is list:
            # Double list.
            rtn = []
            for input in inputs:
                assert (len(input) == 2)
                rtn.append(self._call_one_pair(input))
            return rtn
        else:
            assert (len(inputs) == 2)
            return self._call_one_pair(inputs)

    def _call_one_pair(self, input):
        raise NotImplementedError()


class Dot(Merge):
    """ Dot layer. """

    def __init__(self, output_dim, **kwargs):
        super(Dot, self).__init__(**kwargs)
        self.output_dim = output_dim
        assert (output_dim == 1 or output_dim == 2)

    def _call_one_pair(self, input):
        x_1 = input[0]
        x_2 = input[1]
        assert (x_1.shape == x_2.shape)
        assert (x_1.shape[0] == 1)
        emb_dim = x_1.get_shape().as_list()[1]

        # one pair comparison
        sim_score = interact_two_sets_of_vectors(
            x_1, x_2, 1,  # interact only once
            W=[tf.eye(emb_dim)],
            act=tf.sigmoid)  # use sigmoid to squash the score between (0, 1)

        if self.output_dim == 2:
            output = tf.concat([sim_score, 1 - sim_score], 1)
            assert (output.shape == (1, 2))
        else:
            assert (self.output_dim == 1)
            output = tf.reshape(sim_score, [1, 1])
        return output


class SLM(Merge):
    """ Single Layer model.
    (Socher, Richard, et al.
    "Reasoning with neural tensor networks for knowledge base completion."
    NIPS. 2013.). """

    def __init__(self, input_dim, output_dim, act, dropout, bias, **kwargs):
        super(SLM, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.act = act
        self.bias = bias
        with tf.variable_scope(self.name + '_vars'):
            self.vars['V'] = glorot([self.output_dim, input_dim * 2], name='V')
            if self.bias:
                self.vars['b'] = zeros([output_dim], name='b')
        self.handle_dropout(dropout)
        self._log_vars()

    def _call_one_pair(self, input):
        x_1 = input[0]
        x_2 = input[1]

        # dropout
        x_1 = tf.nn.dropout(x_1, 1 - self.dropout)
        x_2 = tf.nn.dropout(x_2, 1 - self.dropout)

        # one pair comparison
        return interact_two_sets_of_vectors(
            x_1, x_2, self.output_dim,
            V=self.vars['V'],
            b=self.vars['b'] if self.bias else None,
            act=self.act)


class NTN(Merge):
    """ NTN layer.
    (Socher, Richard, et al.
    "Reasoning with neural tensor networks for knowledge base completion."
    NIPS. 2013.). """

    def __init__(self, input_dim, feature_map_dim, apply_u, dropout,
                 inneract, bias, **kwargs):
        super(NTN, self).__init__(**kwargs)

        self.feature_map_dim = feature_map_dim
        self.apply_u = apply_u
        self.bias = bias
        self.inneract = inneract

        self.handle_dropout(dropout)

        with tf.variable_scope(self.name + '_vars'):
            self.vars['V'] = glorot([feature_map_dim, input_dim * 2], name='V')
            self.vars['W'] = glorot([feature_map_dim, input_dim, input_dim],
                                    name='W')
            if self.bias:
                self.vars['b'] = zeros([feature_map_dim], name='b')
            if self.apply_u:
                self.vars['U'] = glorot([feature_map_dim, 1], name='U')

        self._log_vars()

    def _call_one_pair(self, input):
        x_1 = input[0]
        x_2 = input[1]

        # dropout
        x_1 = tf.nn.dropout(x_1, 1 - self.dropout)
        x_2 = tf.nn.dropout(x_2, 1 - self.dropout)

        # one pair comparison
        return interact_two_sets_of_vectors(
            x_1, x_2, self.feature_map_dim,
            V=self.vars['V'],
            W=self.vars['W'],
            b=self.vars['b'] if self.bias else None,
            act=self.inneract,
            U=self.vars['U'] if self.apply_u else None)


def interact_two_sets_of_vectors(x_1, x_2, interaction_dim, V=None,
                                 W=None, b=None, act=None, U=None):
    """
    Calculates the interaction scores between each row of x_1 (a marix)
        and x_2 ( a vector).
    :param x_1: an (N, D) matrix (if N == 1, a (1, D) vector)
    :param x_2: a (1, D) vector
    :param interaction_dim: number of interactions
    :param V:
    :param W:
    :param b:
    :param act:
    :param U:
    :return: if U is not None, interaction results of (N, interaction_dim)
             if U is None, interaction results of (N, 1)
    """
    feature_map = []
    for i in range(interaction_dim):
        middle = 0.0
        if V is not None:
            # In case x_2 is a vector but x_1 is a matrix, tile x_2.
            tiled_x_2 = tf.multiply(tf.ones_like(x_1), x_2)
            concat = tf.concat([x_1, tiled_x_2], 1)
            v_weight = tf.reshape(V[i], [-1, 1])
            V_out = tf.matmul(concat, v_weight)
            middle += V_out
        if W is not None:
            temp = tf.matmul(x_1, W[i])
            h = tf.matmul(temp, tf.transpose(x_2))  # h = K.sum(temp*e2,axis=1)
            middle += h
        if b is not None:
            middle += b[i]
        feature_map.append(middle)

    output = tf.concat(feature_map, 1)
    if act is not None:
        output = act(output)
    if U is not None:
        output = tf.matmul(output, U)

    return output


class MNE(Merge):
    """ MNE layer. """

    def __init__(self, input_dim, dropout, inneract, **kwargs):
        super(MNE, self).__init__(**kwargs)

        self.inneract = inneract
        self.input_dim = input_dim
        self.handle_dropout(dropout)

    def _call_one_pair(self, input):
        # Assume x_1 & x_2 are of dimension N * D
        x_1 = input[0]
        x_2 = input[1]

        # dropout
        x_1 = tf.nn.dropout(x_1, 1 - self.dropout)
        x_2 = tf.nn.dropout(x_2, 1 - self.dropout)

        # one pair comparison
        output = tf.matmul(x_1, tf.transpose(x_2))

        return self.inneract(output)


class MNEMatch(Layer):
    """ MNE layer. """

    def __init__(self, inneract, **kwargs):
        super(MNEMatch, self).__init__(**kwargs)

        self.inneract = inneract

    def _call(self, inputs):
        if type(inputs) is list:
            rtn = []
            for input in inputs:
                rtn.append(self._call_one_pair(input))
            return rtn
        else:
            return self._call_one_pair(inputs)

    def _call_one_pair(self, input):
        x_1 = input[0]
        x_2 = input[1]

        # Assume no padding layer before MNEMatch
        self.row_num = tf.shape(x_1)[0]
        match_num = tf.minimum(tf.shape(x_1)[0], tf.shape(x_2)[0])
        total_num = tf.maximum(tf.shape(x_1)[0], tf.shape(x_2)[0])
        submat = tf.matmul(x_1, tf.transpose(x_2))
        output = tf.constant([])
        for _ in range(match_num):
            row, col = self._find_max_index(submat)
            output = tf.stack(output, submat[row, col])
            submat = self._remove_row_col(submat, row, col)
        return tf.reduce_sum(output) / total_num

    def _remove_row_col(self, mat, row, col):
        row_idx = tf.concat([tf.range(0, col), tf.range(col + 1, tf.shape(mat)[0])], 0)
        mat = tf.gather(mat, row_idx)
        col_idx = tf.concat([tf.range(0, row), tf.range(row + 1, tf.shape(mat)[0])], 0)
        mat = tf.gather(mat, col_idx, axis=1)
        self.row_num -= 1
        return mat

    def _find_max_index(self, x):
        flat_x = tf.squeeze(tf.reshape(x, [-1, 1]))
        idx = tf.argmax(flat_x)
        row = tf.floordiv(idx, self.row_num)
        col = tf.floormod(idx, self.row_num)
        return row, col


""" ############# End of merging two graph-level embeddings. ############# """


class ANPM(Merge):
    """ Attention_NTN_Padding_MNE layer. """

    def __init__(self, input_dim, att_times, att_num, att_style, att_weight,
                 feature_map_dim, dropout, bias, ntn_inneract, apply_u,
                 padding_value,
                 mne_inneract, mne_method, branch_style, **kwargs):  # num_bins
        super(ANPM, self).__init__(**kwargs)

        if 'an' in branch_style:
            self.att_layer = Attention(input_dim=input_dim,
                                       att_times=att_times,
                                       att_num=att_num,
                                       att_style=att_style,
                                       att_weight=att_weight)

            self.ntn_layer = NTN(
                input_dim=input_dim * att_num,
                feature_map_dim=feature_map_dim,
                dropout=dropout,
                inneract=ntn_inneract,
                apply_u=apply_u,
                bias=bias)

        if 'pm' in branch_style:
            self.padding_layer = Padding(padding_value=padding_value)

            self.mne_layer = MNE(
                input_dim=input_dim,
                dropout=dropout,
                inneract=mne_inneract)

        if 'hist_' in mne_method:  # e.g. hist_16_norm, hist_32_raw
            self.num_bins = self._get_mne_output_dim(mne_method)

        self.mne_method = mne_method
        self.branch = branch_style

    def produce_graph_level_emb(self):
        return 'an' in self.branch

    def produce_node_atts(self):
        return 'an' in self.branch

    def merge_graph_level_embs(self):
        return True

    def _call_one_pair(self, input):
        x_1 = input[0]
        x_2 = input[1]
        assert (x_1.shape[1] == x_2.shape[1])
        sim_scores, mne_features = self._get_ntn_scores_mne_features(x_1, x_2)

        # Merge.
        if self.branch == 'an':
            output = sim_scores
        elif self.branch == 'pm':
            output = mne_features
        elif self.branch == 'anpm':
            to_concat = [sim_scores, mne_features]
            output = tf.concat(to_concat, axis=1)
            # output = tf.nn.l2_normalize(output)
        else:
            raise RuntimeError('Unknown branching style {}'.format(self.branch))

        return output

    def _get_ntn_scores_mne_features(self, x_1, x_2):
        sim_scores, mne_features = None, None
        # Branch 1: Attention + NTN.
        if 'an' in self.branch:
            x_1_gemb = self.att_layer(x_1)
            x_2_gemb = self.att_layer(x_2)
            self.embeddings = [x_1_gemb, x_2_gemb]
            self.att = self.att_layer.att
            sim_scores = self.ntn_layer([x_1_gemb, x_2_gemb])
        # Branch 2: Padding + MNE.
        if 'pm' in self.branch:
            mne_features = self._get_mne_features(x_1, x_2)
        return sim_scores, mne_features

    def _get_mne_features(self, x_1, x_2):
        max_dim = tf.maximum(tf.shape(x_1)[0], tf.shape(x_2)[0])
        x_1_pad = self.padding_layer(x_1)
        x_2_pad = self.padding_layer(x_2)
        mne_mat = self.mne_layer([x_1_pad, x_2_pad])
        if 'hist_' in self.mne_method:
            mne_mat = mne_mat[:max_dim, :max_dim]
            x = tf.reshape(
                tf.histogram_fixed_width(mne_mat, value_range=[0, 1],
                                         nbins=self.num_bins),
                [1, -1])
            x /= tf.reduce_sum(x)  # L1-normalize
        elif self.mne_method == 'arg_max_naive':
            x_row = tf.reduce_max(mne_mat, reduction_indices=[0])
            x_col = tf.reduce_max(mne_mat, reduction_indices=[1])
            x = tf.reshape(tf.concat([x_row, x_col], 0), [1, -1])
        else:
            raise 'Unknown mne method {}'.format(self.mne_method)
        output = tf.cast(x, tf.float32)
        return output

    def _get_mne_output_dim(self, mne_method):
        if 'hist_' in mne_method:  # e.g. hist_16_norm, hist_32_raw
            ss = mne_method.split('_')
            assert (ss[0] == 'hist')
            rtn = int(ss[1])
        else:
            raise NotImplementedError()
        return rtn


class ANPMD(ANPM):
    """ Attention_NTN_Padding_MNE_Dense_Two_Branch layer. """

    def __init__(self, input_dim, att_times, att_num, att_style, att_weight,
                 feature_map_dim, dropout, bias, ntn_inneract, apply_u,
                 padding_value,
                 mne_inneract, mne_method, branch_style,
                 dense1_dropout, dense1_act, dense1_bias, dense1_output_dim,
                 dense2_dropout, dense2_act, dense2_bias, dense2_output_dim,
                 **kwargs):  # num_bins
        super(ANPMD, self).__init__(
            input_dim, att_times, att_num, att_style, att_weight,
            feature_map_dim, dropout, bias, ntn_inneract, apply_u,
            padding_value,
            mne_inneract, mne_method, branch_style, **kwargs)
        if 'hist_' in self.mne_method:
            pm_dense_1_input_dim = self.num_bins
        elif self.mne_method == 'arg_max_naive':
            pm_dense_1_input_dim = 2 * FLAGS.max_nodes
        else:
            raise 'Unknown method {}'.format(self.mne_method)

        # Dense layer for NTN
        if 'an' in self.branch:
            self.an_dense_1 = Dense(input_dim=feature_map_dim,
                                    output_dim=dense1_output_dim,
                                    dropout=dense1_dropout,
                                    act=dense1_act,
                                    bias=dense1_bias)

            self.an_dense_2 = Dense(input_dim=dense1_output_dim,
                                    output_dim=dense2_output_dim,
                                    dropout=dense2_dropout,
                                    act=dense2_act,
                                    bias=dense2_bias)

        # Dense layer for MNE
        if 'pm' in self.branch:
            self.pm_dense_1 = Dense(input_dim=pm_dense_1_input_dim,
                                    output_dim=dense1_output_dim,
                                    dropout=dense1_dropout,
                                    act=dense1_act,
                                    bias=dense1_bias)

            self.pm_dense_2 = Dense(input_dim=dense1_output_dim,
                                    output_dim=dense2_output_dim,
                                    dropout=dense2_dropout,
                                    act=dense2_act,
                                    bias=dense2_bias)

    def _call_one_pair(self, input):
        x_1 = input[0]
        x_2 = input[1]
        assert (x_1.shape[1] == x_2.shape[1])
        sim_scores, mne_features = self._get_ntn_scores_mne_features(x_1, x_2)

        # Merge.
        an_output, pm_output = None, None
        if 'an' in self.branch:
            an_output = self.an_dense_1(sim_scores)
            an_output = self.an_dense_2(an_output)
        if 'pm' in self.branch:
            pm_output = self.an_dense_1(mne_features)
            pm_output = self.an_dense_2(pm_output)
        if self.branch == 'an':
            output = an_output
        elif self.branch == 'pm':
            output = pm_output
        else:
            assert (self.branch == 'anpm')
            to_concat = [an_output, pm_output]
            output = tf.concat(to_concat, axis=1)

        return output


class ANNH(ANPM):
    """ Attention_NTN_Node_Histogram layer. """

    def __init__(self, input_dim, att_times, att_num, att_style, att_weight,
                 feature_map_dim, dropout, bias, ntn_inneract, apply_u,
                 padding_value,
                 mne_inneract, mne_method, branch_style, **kwargs):  # num_bins
        super(ANNH, self).__init__(input_dim, att_times, att_num, att_style, att_weight,
                                   feature_map_dim, dropout, bias, ntn_inneract, apply_u,
                                   padding_value,
                                   mne_inneract, mne_method, branch_style, **kwargs)
        assert (branch_style == 'an' or branch_style == 'anpm')

        if branch_style == 'anpm':
            pm_input_dim = None
            if 'hist_' in mne_method:
                pm_input_dim = self.num_bins
            else:
                raise NotImplementedError()
            self.ntn_layer_for_pm = NTN(
                input_dim=pm_input_dim,
                feature_map_dim=feature_map_dim,
                dropout=dropout,
                inneract=ntn_inneract,
                apply_u=apply_u,
                bias=bias)

    def _call_one_pair(self, input):
        x_1 = input[0]
        x_2 = input[1]
        assert (x_1.shape[1] == x_2.shape[1])
        # Branch 1: Attention + NTN.
        x_1_gemb = self.att_layer(x_1)
        x_2_gemb = self.att_layer(x_2)
        self.embeddings = [x_1_gemb, x_2_gemb]
        self.att = self.att_layer.att
        to_ntn = [x_1_gemb, x_2_gemb]
        sim_scores = self.ntn_layer(to_ntn)
        # Branch 2: Padding + MNE.
        if 'pm' in self.branch:
            mne_features_1 = self._get_mne_features(x_1, x_1)
            mne_features_2 = self._get_mne_features(x_2, x_2)
            mne_feat_interact = self.ntn_layer_for_pm(
                [mne_features_1, mne_features_2])
            sim_scores = tf.concat([sim_scores, mne_feat_interact], axis=1)
        return sim_scores


class CNN(Layer):
    """ CNN layer. """

    def __init__(self, start_cnn, end_cnn, window_size, kernel_stride, in_channel,
                 out_channel, padding, pool_size, dropout, act, bias, **kwargs):
        super(CNN, self).__init__(**kwargs)

        self.filter = [window_size, window_size, in_channel, out_channel]
        self.stride = kernel_stride
        self.padding = padding
        self.pool_size = pool_size
        self.bias = bias
        self.act = act
        self.out_channel = out_channel
        self.start_cnn = start_cnn
        self.end_cnn = end_cnn

        self.handle_dropout(dropout)

        with tf.variable_scope(self.name + '_vars'):
            self.vars['kernels'] = xavier(self.filter, self.name)
            # self.vars['kernels'] = glorot(self.filter, self.name)
            if self.bias:
                self.vars['bias'] = zeros([out_channel], name='bias')

        self._log_vars()

    def _call(self, inputs):
        if type(inputs) is list:
            rtn = []
            for input in inputs:
                rtn.append(self._call_one_mat(input))
            return rtn
        else:
            return self._call_one_mat(inputs)

    def _call_one_mat(self, inputs):
        x = inputs
        if self.start_cnn:
            ins_dim = len(x.get_shape().as_list())
            if ins_dim == 2:
                batch_size, row, col = 1, tf.shape(x)[0], tf.shape(x)[1]
            elif ins_dim == 3:
                batch_size, row, col = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
            else:
                raise RuntimeError('CNN dimension not right {}'.format(x.get_shape().as_list()))
        else:
            batch_size, row, col = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        # dropout
        x = tf.nn.dropout(x, 1 - self.dropout)

        input_x = tf.reshape(x, [batch_size, row, col, 1]) if self.start_cnn else x

        # Convolution & Pooling
        if self.bias:
            conv_x = self.act(tf.nn.conv2d(input_x, self.vars['kernels'], strides=[1, self.stride, self.stride, 1],
                                           padding=self.padding) + self.vars['bias'])
        else:
            conv_x = self.act(tf.nn.conv2d(input_x, self.vars['kernels'], strides=[1, self.stride, self.stride, 1],
                                           padding=self.padding))
        pool_x = tf.nn.max_pool(conv_x, ksize=[1, self.pool_size, self.pool_size, 1],
                                strides=[1, self.pool_size, self.pool_size, 1], padding=self.padding)
        # norm_x = tf.nn.lrn(pool_x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        return pool_x if not self.end_cnn else tf.reshape(pool_x, [1, self.out_channel])
        # tf.reshape(pool_x, [1, self.out_channel])


# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}
_LAYERS = []


def get_layer_name(layer):
    """ Helper function, assigns layer names and unique layer IDs."""
    layer_name = layer.__class__.__name__.lower()
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        layer_id = 1
    else:
        _LAYER_UIDS[layer_name] += 1
        layer_id = _LAYER_UIDS[layer_name]
    _LAYERS.append(layer)
    return str(len(_LAYERS)) + '_' + \
           layer_name + '_' + str(layer_id)


def sparse_dropout(x, keep_prob, noise_shape):
    """ Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)
