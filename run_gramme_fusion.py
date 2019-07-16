"""
This python file performs semi-supervised node classification on a multi-layer graph data.
The architecture used in this file is GrAMME-Fusion.
Please see the readme file in the github repo for additional details.
"""

import time
import tensorflow as tf
import argparse
import os

from lib.utils import *
from lib.models import GRAMME_Fusion
from lib.metrics import masked_softmax_cross_entropy, masked_accuracy, regularization_loss


def build_multiple_AH_plus_multiple_FH_model(edgelists_L,
                                             num_nodes,
                                             num_classes,
                                             num_heads,
                                             num_sup_attention,
                                             inp_dim,
                                             out_dim,
                                             learn_rate=0.01,
                                             lambd_reg=0.005):
    # Placeholders
    place_holders = {
        'edgelists_L': [tf.placeholder(tf.int32, edgelist.shape) for edgelist in edgelists_L],  # Edge list
        'input_features': tf.placeholder(tf.float32, [num_nodes, inp_dim]),  # Features : N x F
        'labels': tf.placeholder(tf.float32, shape=[num_nodes, num_classes]),  # True Labels
        'labels_mask': tf.placeholder(dtype=tf.int32),  # Labels mask
        'attention_dropout_keep': tf.placeholder_with_default(1.0, shape=()),
        'input_dropout_keep': tf.placeholder_with_default(1.0, shape=())
    }

    Layer1, Layer1_Linear_Weights, Layer1_Attention_kernels, Layer1_branched_alpha = \
        GRAMME_Fusion(input_h=tf.nn.dropout(x=place_holders['input_features'],keep_prob=place_holders['input_dropout_keep']),
                              input_dim=inp_dim,
                              N=num_nodes,
                              edgelist_List=place_holders['edgelists_L'],
                              output_dim=out_dim,
                              no_attention_heads=num_heads,
                              no_supra_attentions = num_sup_attention,
                              attention_dropout_keep=place_holders['attention_dropout_keep'],
                              act=tf.nn.relu,
                              name='multiple_AH_multiple_FH_Layer_1')

    Layer2, Layer2_Linear_Weights, Layer2_Attention_kernels, Layer2_branched_alpha = \
        GRAMME_Fusion(input_h=tf.nn.dropout(x=Layer1, keep_prob=place_holders['input_dropout_keep']),
                              input_dim=out_dim,
                              N=num_nodes,
                              edgelist_List=place_holders['edgelists_L'],
                              output_dim=num_classes,
                              no_attention_heads=num_heads,
                              no_supra_attentions=num_sup_attention,
                              attention_dropout_keep=place_holders['attention_dropout_keep'],
                              act=None,
                              name='multiple_AH_multiple_FH_Layer_2')

    supervised_loss = masked_softmax_cross_entropy(Layer2, place_holders['labels'], place_holders['labels_mask'])
    regularized_loss = regularization_loss(Layer1_Linear_Weights,lambd_reg) + regularization_loss(Layer2_Linear_Weights,lambd_reg) + \
    regularization_loss(Layer1_Attention_kernels) + regularization_loss(Layer2_Attention_kernels) #+ \
    # regularization_loss([Layer1_branched_alpha, Layer2_branched_alpha])

    loss = supervised_loss + regularized_loss

    y_pred = tf.nn.softmax(Layer2)
    # accuracy = masked_accuracy(y_pred, place_holders['labels'], place_holders['labels_mask'])
    train_op = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss)

    return place_holders, train_op, y_pred, loss, Layer1


def main(args):

    parser = argparse.ArgumentParser()
    parser.add_argument('--multiplex_edges_filename', default='data/leskovec_ng/Leskovec-Ng.multilayer.edges')
    parser.add_argument('--multiplex_labels_filename', default='data/leskovec_ng/Leskovec-Ng.multilayer.labels')
    parser.add_argument('--multiplex_features_filename', default=None)
    parser.add_argument('--train_percentage', default=10)
    parser.add_argument('--random_seed', default=1)
    parser.add_argument('--learning_rate', default=0.01)

    args = parser.parse_args()

    train_script = os.path.basename(__file__)
    multiplex_edges_filename = args.multiplex_edges_filename
    multiplex_labels_filename = args.multiplex_labels_filename
    multiplex_features_filename = args.multiplex_features_filename
    seed = int(args.random_seed)
    train_percent = int(args.train_percentage)
    learn_rate = args.learning_rate

    file_name = 'results.txt'

    if os.path.exists(file_name):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not


    file = open(file_name, append_write)

    file.write("*******************************************************\n")
    file.write("Multiplex Edgelist: " + multiplex_edges_filename + "\n")
    file.write("Train nodes %: " + str(train_percent) + "\n")

    # Set random seed
    np.random.seed(seed)
    tf.set_random_seed(seed)

    edgelists_L, multiplex_features, y_train, y_val, y_test, \
    train_mask, val_mask, test_mask, y_labels = load_dataset(multiplex_edges_filename,
                                                             multiplex_labels_filename,
                                                             multiplex_features_filename,
                                                             train_percent=train_percent,
                                                             seed=seed)
    print('Data Preprocessing Done')

    # Model building parameters.
    N = len(y_labels)  # number of nodes in the graph
    P = len(edgelists_L)  # number of layers(relations) in the graphs
    num_heads = 2
    num_sup_attention = 5
    num_classes = y_train.shape[1]
    inp_dim = 16
    out_dim_layer1 = 8
    att_dropout_keeprate = 0.6
    inp_dropout_keeprate = 0.6

    # Training parameters
    num_epochs = 200
    lambd_reg = 0.005

    # Model parameters
    file.write("Model Parameters - "+"\n")
    file.write("Number of Attention Heads: {}\n".format(num_heads))
    file.write("Number of Supra Attentions: {}\n".format(num_sup_attention))


    # Generate random features.
    if multiplex_features is None:
        multiplex_features = np.random.randn(N, inp_dim)
    # TODO: now nodes across layers have the same feature, do we need to generate different ones?


    place_holders, train_op, pred, loss, Layer1 = build_multiple_AH_plus_multiple_FH_model(edgelists_L, N, num_classes, num_heads,
                                                                                           num_sup_attention, inp_dim, out_dim_layer1, learn_rate=learn_rate,
                                                              lambd_reg=lambd_reg)

    print('Model Setup')
    feed_dict_train = construct_Baseline_feed_dict(edgelists_L, multiplex_features, y_train, train_mask, place_holders,
                                                   attention_dropout=att_dropout_keeprate,
                                                   input_dropout=inp_dropout_keeprate)
    feed_dict_val = construct_Baseline_feed_dict(edgelists_L, multiplex_features, y_val, val_mask, place_holders)
    feed_dict_test = construct_Baseline_feed_dict(edgelists_L, multiplex_features, y_test, test_mask, place_holders)

    # Train model and perform testing.
    with tf.Session() as sess:
        # Init variables
        sess.run(tf.global_variables_initializer())
        print('Training Starts')
        train_losses, train_accs, val_losses, val_accs = [], [], [], []
        Layer1_features_L = []  # Stores hidden representations.

        for epoch in range(1,num_epochs+1):
            t = time.time()
            _, train_loss, train_pred = sess.run([train_op, loss, pred], feed_dict=feed_dict_train)
            train_losses.append(train_loss)
            train_acc = masked_accuracy(train_pred, y_train, train_mask)
            train_accs.append(train_acc)

            # Validation
            val_loss, val_pred = sess.run([loss, pred], feed_dict=feed_dict_val)
            val_losses.append(val_loss)
            val_acc = masked_accuracy(val_pred, y_val, val_mask)
            val_accs.append(val_acc)

            # Print results every 10 epochs
            if epoch % 10 == 0:
                print("Epoch:", '%04d' % (epoch),
                      "train_loss=", "{:.5f}".format(train_loss), "train_acc=", "{:.5f}".format(train_acc),
                      "time=", "{:.5f}".format((time.time() - t)*10))

            Layer1_features_L.append(sess.run(Layer1, feed_dict=feed_dict_test))

        print("Optimization Finished!")

        # Testing
        t_test = time.time()
        test_loss, test_pred = sess.run([loss, pred], feed_dict=feed_dict_test)
        test_acc = masked_accuracy(test_pred, y_test, test_mask)
        print("Test set results:", "cost=", "{:.5f}".format(test_loss),
              "accuracy=", "{:.5f}".format(max(val_accs)), "Test time=", "{:.5f}".format(time.time() - t_test))

    file.write("GrAMME Fusion Accuracy: " + str(max(val_accs)) + "\n")
    file.close()

if __name__ == '__main__':
    main(sys.argv)
