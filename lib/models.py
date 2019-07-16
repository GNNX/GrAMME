import tensorflow as tf
import numpy as np
from lib.metrics import regularization_loss
from lib.utils import flatten_lists


def GRAMME_SG(features=None,  # N x F
                            input_dim=None,
                            num_nodes=None,
                            num_layers=None,
                            all_edges=None,
                            output_dim=2,
                            no_attention_heads=1,
                            attention_dropout_keep=1,
                            act=None,
                            name='Default'
                            ):
    """
    This is the function for implementing GraMME-SG. It takes supra features, all edges as its input

    features                    -- placeholder/tensor - Feature matrix of size NxF (no of nodes x feature dimension)
    input_dim                   -- scalar, input dimension, or the feature dimension, F
    total_nodes                 -- scalar, NP
    all_edges                   -- placeholder/tensor for edgelist consisting of  all edges
    output_dim                  -- scalar, output dimension for the GAT layer, F'
    no_attention_heads          -- scalar, number of attention heads inside the GAT layer, K
    attention_dropout_keep      -- scalar, dropout keep probability for attentions
    act                         -- tensorflow activation function
    name                        -- name for the baseline Multilayer GAT layer


    Returns:

    Final_Output        -- tensor, output of the BaselineMulitLayerGAT layer
    Linear_Weights      -- python list of Linear weights of the BaselineGAT layer, each entry in the list is a tf Varibale,
                           one linear weight per head (or per graph layer), size input_dim x output_dim, (FxF')
    Attention_kernels   -- python list of attention weights of the GAT layer, each entry in the list is a tf Varibale
                           one attention weight per head, size output_dim x 1, (F'x1) per entry in tuple
    branch_alpha        -- branching weights for the multihead, it is a tf Variable of size no_attention_heads x 1 (Kx1)
        """

    Linear_Weights, Attention_kernels, Outputs = [], [], []

    supra_features = tf.tile(features,  [num_layers, 1])

    with tf.variable_scope(name):
        # branch_alpha = tf.Variable(tf.truncated_normal([no_attention_heads, 1], stddev=0.1),
        #                            name='attention_branched')
        branch_alpha = tf.get_variable(name='Attention_Branch', shape=[no_attention_heads, 1],
                                       regularizer=tf.contrib.layers.l1_regularizer(0.001), constraint=None)

    for head in range(no_attention_heads):
        W = glorot([input_dim, output_dim], name='weights_' + str(head))

        attention0 = tf.Variable(tf.truncated_normal([output_dim, 1], stddev=0.1), name='attention0_' + str(head))
        attention1 = tf.Variable(tf.truncated_normal([output_dim, 1], stddev=0.1), name='attention1_' + str(head))
        att = (attention0, attention1)

        Linear_Weights.append(W)
        Attention_kernels.append(att)

        node_list_i = all_edges[:, 0]
        node_list_j = all_edges[:, 1]

        h_dash = simple_attention_head(h_inp=supra_features,
                                       W=W,
                                       num_nodes=num_nodes*num_layers,
                                       attention=att,
                                       node_list_i=node_list_i,
                                       node_list_j=node_list_j,
                                       keep_dropout=attention_dropout_keep
                                       )

        if act is not None:  # compute activations if required
            h_dash_activated = act(h_dash)
            Outputs.append(h_dash_activated)
        else:
            Outputs.append(h_dash)

        print(name + ' : ' + 'head ' + str(head))

    stacked_outputs = tf.stack(values=Outputs)

    # branch_alpha = tf.nn.softmax(branch_alpha, dim=0 )

    all_nodes_Output = tf.squeeze(tf.tensordot(stacked_outputs, branch_alpha, axes=[[0], [0]]))
    nodes_features = tf.split(value=all_nodes_Output,num_or_size_splits=num_layers,axis=0)
    Final_Output =  tf.reduce_mean(tf.stack(nodes_features),axis=0)

    return Final_Output, Linear_Weights, Attention_kernels, branch_alpha


def GRAMME_Fusion(input_h=None,  #
                                     input_dim=None,
                                     N=None,
                                     edgelist_List=None,
                                     output_dim=10,
                                     no_attention_heads=1,
                                     no_supra_attentions=1,
                                     attention_dropout_keep=1,
                                     act=None,
                                     name='Default'
                                     ):
    """
    This function implements GraMME-Fusion. Here one graph layer (edgelist) is used for one independent Branched GAT layer
    and final branching is performed across the branched GATs (late fusion).That is, each edgelist is independently run for
    a number of heads with branching.

    input_h                     -- placeholder/tensor - Feature matrix of size NxF (no of nodes x feature dimension)
    input_dim                   -- scalar, input dimension, or the feature dimension, F
    N                           -- scalar, number of nodes in the graph, N
    edgelist_List               -- python list of placeholder/tensor
    output_dim                  -- scalar, output dimension for the GAT layer, F'
    no_attention_heads          -- scalar, number of attention heads inside the GAT layer, K
    attention_dropout_keep      -- scalar, dropout keep probability for attentions
    act                         -- tensorflow activation function
    name                        -- name for the baseline Multilayer GAT layer


    Returns:

    Final_Output        -- tensor, output of the BaselineMultiLayerIndependentGAT layer
    Linear_Weights      -- python list of Linear weights of the BaselineMultiLayerIndependentGAT layer, each entry in the list is a tf Varibale,
                           one linear weight per head (or per graph layer), size input_dim x output_dim, (FxF')
    Attention_kernels   -- python list of attention weights of the GAT layer, each entry in the list is a tf Varibale
                           one attention weight per head, size output_dim x 1, (F'x1) per entry in tuple
    Branched_Alpha        -- branching weights for the multihead per layer
    """
    num_layers = len(edgelist_List)
    layer_d, linear_weights_d, attention_kernels_d, branched_Alpha_d = {}, {}, {}, {}
    Linear_Weights, Attention_kernels, Branched_Alpha, Supra_Attentions, Outputs = [], [], [], [], []

    with tf.variable_scope(name):
        # branch_alpha = tf.Variable(tf.truncated_normal([no_attention_heads, 1], stddev=0.1), name='attention_branched')
        final_branch_attention = tf.get_variable(name='Final_Attention_Branch_', shape=[no_supra_attentions, 1],
                                                 regularizer=tf.contrib.layers.l1_regularizer(0.001), constraint=None)

    for i in range(no_supra_attentions):
        with tf.variable_scope(name):
            # branch_alpha = tf.Variable(tf.truncated_normal([no_attention_heads, 1], stddev=0.1), name='attention_branched')
            supra_attention = tf.get_variable(name='Supra_Attention_Branch_' + str(i), shape=[num_layers, 1],
                                              regularizer=tf.contrib.layers.l1_regularizer(0.001), constraint=None)
        Supra_Attentions.append(supra_attention)


    for l in range(num_layers):
        layer_d['layer' + str(l)], linear_weights_d['layer' + str(l)], attention_kernels_d['layer' + str(l)], \
        branched_Alpha_d['layer' + str(l)] = BranchedGATLayerSparse(
            input_h=input_h,
            input_dim=input_dim,
            N=N,
            edge_list_tensor=edgelist_List[l],
            output_dim=output_dim,
            no_attention_heads=no_attention_heads,
            attention_dropout_keep=attention_dropout_keep,
            act=tf.nn.relu,
            name=name + '_' + str(l))

        Outputs.append(layer_d['layer' + str(l)])
        Linear_Weights.append(linear_weights_d['layer' + str(l)])
        Attention_kernels.append(attention_kernels_d['layer' + str(l)])
        Branched_Alpha.append(branched_Alpha_d['layer' + str(l)])

    stacked_outputs = tf.stack(values=Outputs)
    Branched_Alpha.append(final_branch_attention)


    Supra_Outputs = []

    for i in range(no_supra_attentions):
        sup_att = Supra_Attentions[i]
        supra_output = tf.squeeze(tf.tensordot(stacked_outputs, sup_att, axes=[[0], [0]]))

        Supra_Outputs.append(supra_output)

    Final_Outputs_stacked = tf.stack(values=Supra_Outputs)

    Final_Output = tf.squeeze(tf.tensordot(Final_Outputs_stacked, final_branch_attention, axes=[[0], [0]]))

    return Final_Output, flatten_lists(Linear_Weights), flatten_lists(Attention_kernels), Branched_Alpha

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def GraphAttentionLayerSparse(input_h=None,  #
                              input_dim=None,
                              edgelist=None,
                              num_nodes=None,
                              output_dim=10,
                              no_attention_heads=1,
                              attention_reduction_type='concat',
                              attention_dropout_keep=1,
                              act=None
                              ):
    """ This function implements a Basic and sparse Graph Attention Network Layer, uses an edgelist instead of an adjacency matrix.
        This is a sparse implementation and requires less Memory but more time as tensorflow operations are not optimal for sparse implementation.

        The function takes in the following parameters as

        Input arguments:

        input_h                 -- placeholder/tensor - Feature matrix of size NxF (no of nodes x feature dimension)
        input_dim               -- scalar, input dimension, or the feature dimension, F
        edgelist                -- placeholder/tensor - Edgelist information about the graph
        num_nodes               -- scalar, number of nodes in the graph, N
        output_dim              -- scalar, output dimension for the GAT layer, F'
        no_attention_heads      -- scalar, number of attention heads inside the GAT layer, K
        attention_dropout_keep  -- scalar, dropout keep probability for attentions
        act                     -- tensorflow activation function


        Returns:

        Final_Output        -- tensor, output of the branched GAT layer
        Linear_Weights      -- python list of Linear weights of the GAT layer, each entry in the list is a tf Varibale,
                               one linear weight per head, size input_dim x output_dim, (FxF')
        Attention_kernels   -- python list of attention weights of the GAT layer, each entry in the list is a tf Varibale
                               one attention weight per head, size output_dim x 1, (F'x1) per entry in tuple
        branch_alpha        -- branching weights for the multihead, it is a tf Variable of size no_attention_heads x 1 (Kx1)
    """

    Linear_Weights = []
    Outputs = []

    node_list_i = edgelist[:, 0]
    node_list_j = edgelist[:, 1]

    for i in range(no_attention_heads):
        W = glorot([input_dim, output_dim], name='weights_' + str(i))  # Linear weights, F x F'
        # attention a consists of two parts, F' x 1 and F' x 1
        a0 = tf.Variable(tf.truncated_normal([output_dim, 1], stddev=0.1), name='attentions_' + str(i))
        a1 = tf.Variable(tf.truncated_normal([output_dim, 1], stddev=0.1), name='attentions_' + str(i))

        Linear_Weights.append(W)

        # Linear Transformation
        h_tilda = tf.matmul(input_h, W)  # N x F'

        # Attention dot product on both parts.
        att_i = tf.squeeze(tf.matmul(h_tilda, a0, ))  # (N, )
        att_j = tf.squeeze(tf.matmul(h_tilda, a1))  # (N, )

        # Obtain both parts according to edgelist. Will be added up later.
        att_i_expanded = tf.gather(att_i, node_list_i)  # (M, ), M is number of edges
        att_j_expanded = tf.gather(att_j, node_list_j)  # (M, )

        # Partition attention representations for j's to get the summation for each i.
        partitions = tf.dynamic_partition(tf.exp(tf.add(att_i_expanded, att_j_expanded)), node_list_i, num_nodes)
        sum_i = tf.convert_to_tensor([tf.reduce_sum(partition_i) for partition_i in partitions])  # (N, )
        print('Shape of sum_i: ', sum_i.get_shape())

        # Calculate softmax
        alpha = tf.div(tf.exp(tf.add(att_i_expanded, att_j_expanded)), tf.gather(sum_i, node_list_i))  # (M, )
        print('Shape of alpha: ', alpha.get_shape())

        # Dropout
        dropout_softmax = tf.nn.dropout(alpha, keep_prob=attention_dropout_keep)  # TODO: modify prob?

        # Calculate attention representation - linear combination based on the weights alpha.
        weighted_reps_j = tf.multiply(tf.gather(h_tilda, node_list_j, ),
                                      tf.expand_dims(dropout_softmax, axis=1))  # (M, F')
        rep_partitions = tf.dynamic_partition(weighted_reps_j, node_list_i, num_nodes)
        h_dash = tf.convert_to_tensor([tf.reduce_sum(rep_partition, axis=0) for rep_partition in rep_partitions])

        if attention_reduction_type == 'concat' and act is not None:  # compute activations before concatenating
            Outputs.append(act(h_dash))
        else:
            Outputs.append(h_dash)

    # Aggregate all heads.
    if attention_reduction_type == 'concat':
        Final_Output = tf.concat(values=Outputs, axis=1)  # N x kF'
    else:
        Final_Output = tf.reduce_mean(tf.stack(values=Outputs), axis=0)

    return Final_Output, Linear_Weights


def simple_attention_head(h_inp=None,
                          W=None,
                          num_nodes=None,
                          attention=None,
                          node_list_i=None,
                          node_list_j=None,
                          keep_dropout=1
                          ):
    """ Computes a single attention head. A single attention head takes input features and edgelist along with
    linear weights and attention weights. Performs a linear transformation on the features with W and aggregates
    the features from the neighbourhood according to the learnt attention values.

    The function takes in the following parameters as

    Input arguments:

    h_inp               -- placeholder/tensor - Feature matrix of size NxF (no of nodes x feature dimension)
    W                   -- tf Variable, Linear weight for the head, size input_dim x output_dim, FxF'
    num_nodes           -- Number of nodes in the graph
    attention           -- tuple of tf Variables, attention weights as tuples for sparse implementation
    node_list_i         -- tensor, first column of the edgelist
    node_list_j         -- tensor, second column of the edgelist
    keep_dropout        -- scalar, dropout keep probability for attentions


    Returns:

    h_dash_single_head  -- tensor, output of the single attention head
    """

    h_tilda = tf.matmul(h_inp, W)  # N x F'

    att_0, att_1 = attention

    E_0 = tf.squeeze(tf.matmul(h_tilda, att_0))
    E_1 = tf.squeeze(tf.matmul(h_tilda, att_1))

    E_0_i = tf.gather(params=E_0, indices=node_list_i)
    E_1_j = tf.gather(params=E_1, indices=node_list_j)

    E = tf.squeeze(E_0_i + E_1_j)

    partitions = tf.dynamic_partition(tf.exp(E), node_list_i, num_nodes)

    sum_i = tf.convert_to_tensor([tf.reduce_sum(partition_i) for partition_i in partitions])
    sum_i_denominator = tf.gather(sum_i, node_list_i)

    alpha = tf.div(tf.exp(E), sum_i_denominator)

    alpha_dropout = tf.nn.dropout(alpha, keep_prob=keep_dropout)

    h_tilda_extended = tf.gather(h_tilda, node_list_j, )

    weighted_reps_j = tf.multiply(h_tilda_extended, tf.expand_dims(alpha_dropout, axis=1))  # (M, F')

    rep_partitions = tf.dynamic_partition(weighted_reps_j, node_list_i, num_nodes)
    h_dash_single_head = tf.convert_to_tensor(
        [tf.reduce_sum(rep_partition, axis=0) for rep_partition in rep_partitions])

    return h_dash_single_head


def BranchedGATLayerSparse(input_h=None,  #
                           input_dim=None,
                           N=None,
                           edge_list_tensor=None,
                           output_dim=10,
                           no_attention_heads=1,
                           attention_dropout_keep=1,
                           act=None,
                           name='Default'
                           ):
    """ This function implements a sparse and Branched Graph Attention Network Layer with branching at the output of the multihead, uses an edgelist instead of an adjacency matrix.
        This is a sparse implementation and requires less Memory but more time as tensorflow operations are not optimal for sparse implementation.

        The function takes in the following parameters as

        Input arguments:

        input_h                 -- placeholder/tensor - Feature matrix of size NxF (no of nodes x feature dimension)
        input_dim               -- scalar, input dimension, or the feature dimension, F
        N                       -- scalar, number of nodes in the graph, N
        edge_list_tensor        -- placeholder/tensor - Edgelist information about the graph
        output_dim              -- scalar, output dimension for the GAT layer, F'
        no_attention_heads      -- scalar, number of attention heads inside the GAT layer, K
        attention_dropout_keep  -- scalar, dropout keep probability for attentions
        act                     -- tensorflow activation function
        Name                    -- name for the branched GAT layer

        Returns:

        Final_Output        -- tensor, output of the branched GAT layer
        Linear_Weights      -- python list of Linear weights of the GAT layer, each entry in the list is a tf Varibale,
                               one linear weight per head, size input_dim x output_dim, (FxF')
        Attention_kernels   -- python list of attention weights of the GAT layer, each entry in the list is a tf Varibale
                               one attention weight per head, size output_dim x 1, (F'x1) per entry in tuple
        branch_alpha        -- branching weights for the multihead, it is a tf Variable of size no_attention_heads x 1 (Kx1)
        """

    # python list for W, a, and output per attention head
    Linear_Weights = []
    Attention_kernels = []
    Outputs = []

    node_list_i = edge_list_tensor[:, 0]
    node_list_j = edge_list_tensor[:, 1]

    with tf.variable_scope(name):
        # branch_alpha = tf.Variable(tf.truncated_normal([no_attention_heads, 1], stddev=0.1), name='attention_branched')
        branch_alpha = tf.get_variable(name='Attention_Branch', shape=[no_attention_heads, 1],
                                       regularizer=tf.contrib.layers.l1_regularizer(0.001), constraint=None)

    for i in range(no_attention_heads):
        linear_weight = glorot([input_dim, output_dim], name='weights_' + str(i))

        attention0 = tf.Variable(tf.truncated_normal([output_dim, 1], stddev=0.1), name='attention0_' + str(i))
        attention1 = tf.Variable(tf.truncated_normal([output_dim, 1], stddev=0.1), name='attention1_' + str(i))

        # attention = glorot([2 * output_dim, 1], name='attentions_' + str(i))

        Linear_Weights.append(linear_weight)
        Attention_kernels.append((attention0, attention1))

    for i in range(no_attention_heads):
        W = Linear_Weights[i]  # F x F'
        att = Attention_kernels[i]  # 2F' x 1

        h_dash = simple_attention_head(h_inp=input_h,
                                       W=W,
                                       num_nodes=N,
                                       attention=att,
                                       node_list_i=node_list_i,
                                       node_list_j=node_list_j,
                                       keep_dropout=attention_dropout_keep
                                       )

        if act is not None:  # compute activations before concatenating
            h_dash_activated = act(h_dash)
            Outputs.append(h_dash_activated)
        else:
            Outputs.append(h_dash)

        print(name + ' : ' + 'head ' + str(i))

    stacked_outputs = tf.stack(values=Outputs)

    # branch_alpha = tf.nn.softmax(branch_alpha, dim=0 )

    Final_Output = tf.squeeze(tf.tensordot(stacked_outputs, branch_alpha, axes=[[0], [0]]))

    return Final_Output, Linear_Weights, Attention_kernels, branch_alpha
