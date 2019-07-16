"""This is the SupraGraph approach
We construct a large supra graph where we define pillar edges. """

import time
import tensorflow as tf
import argparse
import pickle
import os

from lib.utils import *
from lib.models import GRAMME_SG
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from metrics import masked_softmax_cross_entropy, masked_accuracy, regularization_loss


def build_supra_model_common_features(all_edgelist, num_nodes, num_layers, num_classes, num_heads, inp_dim, out_dim, learn_rate=0.01,
                      lambd_reg=0.005):
    # Placeholders
    place_holders = {'supra_edgelist': tf.placeholder(tf.int32, all_edgelist.shape),
                     'supra_input_features': tf.placeholder(tf.float32, [num_nodes, inp_dim]),
                     'labels': tf.placeholder(tf.float32, shape=[num_nodes, num_classes]),  # True Labels
                     'labels_mask': tf.placeholder(dtype=tf.int32),  # Labels mask
                     'attention_dropout_keep': tf.placeholder_with_default(1.0, shape=()),
                     'input_dropout_keep': tf.placeholder_with_default(1.0, shape=())
                     }

    Layer1, Layer1_Linear_Weights, Layer1_Attention_kernels, Layer1_branched_alpha = GRAMME_SG(
        features=tf.nn.dropout(x=place_holders['supra_input_features'],
                                     keep_prob=place_holders['input_dropout_keep']),  # N*P x F
        input_dim=inp_dim,
        num_nodes=num_nodes,
        num_layers=num_layers,
        all_edges=place_holders['supra_edgelist'],
        output_dim=out_dim,
        no_attention_heads=num_heads,
        attention_dropout_keep=place_holders['attention_dropout_keep'],
        act=tf.nn.relu,
        name='SupraGAT_Layer_1'
    )

    Layer2, Layer2_Linear_Weights, Layer2_Attention_kernels, Layer2_branched_alpha = GRAMME_SG(
        features=tf.nn.dropout(x=Layer1, keep_prob=place_holders['input_dropout_keep']),  # N*P x F
        input_dim=out_dim,
        num_nodes=num_nodes,
        num_layers=num_layers,
        all_edges=place_holders['supra_edgelist'],
        output_dim=num_classes,
        no_attention_heads=num_heads,
        attention_dropout_keep=place_holders['attention_dropout_keep'],
        act=None,
        name='SupraGAT_Layer_2'
    )

    supervised_loss = masked_softmax_cross_entropy(Layer2, place_holders['labels'],
                                                   place_holders['labels_mask'])
    regularized_loss = regularization_loss(Layer1_Linear_Weights, lambd_reg) + regularization_loss(
        Layer2_Linear_Weights, lambd_reg)  # + \

    loss = supervised_loss + regularized_loss

    y_pred = tf.nn.softmax(Layer2)

    train_op = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss)

    return place_holders, train_op, y_pred, loss, Layer1


def main(args):


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='vickers')
    parser.add_argument('--train_percentage', default=10)
    parser.add_argument('--random_seed', default=20)
    parser.add_argument('--learning_rate', default=0.025)

    args = parser.parse_args()

    train_script = os.path.basename(__file__)
    dataset_name = args.dataset_name
    seed = int(args.random_seed)
    train_percent = int(args.train_percentage)

    # print("File name ran ", train_script)
    # print("Dataset Name: ", dataset_name)
    # print("Train Percentage: ", train_percent)
    # print("Random Seed Given: ",seed)
    # print("\n\n")

    path_to_write = "Results/"
    file_name = path_to_write + train_script + '_dataset_' + dataset_name + \
                '_train_percentage_' + str(train_percent) + '.txt'

    if os.path.exists(file_name):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    file = open(file_name, append_write)

    file.write("*******************************************************\n")
    file.write("Approach: " + train_script + "\n\n")
    file.write("Dataset Name: " + dataset_name + "\n")
    file.write("Train nodes %: " + str(train_percent) + "\n")
    file.write("Random Seed: " + str(seed) + "\n\n")

    # Set random seed
    np.random.seed(seed)
    tf.set_random_seed(seed)

    edgelists_L, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, y_labels = \
        load_multilayer_dataset(dataset_name, train_percent=train_percent, seed=seed)

    # Model building parameters.
    N = len(y_labels)  # number of nodes in the graph
    P = len(edgelists_L)  # number of layers(relations) in the graphs
    total_nodes = N * P  # Total number of nodes in the supra-graph
    num_heads = 2  # Independent parameter, does not depend on the number of layers
    num_classes = y_train.shape[1]
    inp_dim = 64
    out_dim_layer1 = 32
    att_dropout_keeprate = 0.7
    inp_dropout_keeprate = 0.7

    # create one big supra edgelist consisting of pillar edges and in-layer edges
    all_edgelist = create_supragraph_edgelist(edgelists_L, N)

    # Creating one big supragraph features
    features = np.random.randn(N, inp_dim)  # Randomly generated features
    # In this case, we assume each node across different layers has same features

    # Training parameters
    num_epochs = 200
    learn_rate = 0.01
    lambd_reg = 0.005

    place_holders, train_op, pred, loss, Layer1 = build_supra_model_common_features(all_edgelist, N, P, num_classes, num_heads,
                                                                    inp_dim, out_dim_layer1, learn_rate=learn_rate,
                                                                    lambd_reg=lambd_reg)
    print('Model Setup')

    # feed dictionaries
    feed_dict_train = construct_Supra_feed_dict(all_edgelist, features, y_train, train_mask, place_holders,
                                                attention_dropout=att_dropout_keeprate, input_dropout=inp_dropout_keeprate)
    feed_dict_val = construct_Supra_feed_dict(all_edgelist, features, y_val, val_mask, place_holders)
    feed_dict_test = construct_Supra_feed_dict(all_edgelist, features, y_test, test_mask, place_holders)

    # Train model and perform testing.
    with tf.Session() as sess:
        # Init variables
        sess.run(tf.global_variables_initializer())
        print('Training Starts')
        train_losses, train_accs, val_losses, val_accs = [], [], [], []
        Layer1_features_L = []  # Stores hidden representations.

        # Train model
        for epoch in range(num_epochs):
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

            # Print results
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(train_loss), "train_acc=", "{:.5f}".format(train_acc),
                  "val_loss=", "{:.5f}".format(val_loss), "val_acc=", "{:.5f}".format(val_acc),
                  "time=", "{:.5f}".format(time.time() - t))

            Layer1_features_L.append(sess.run(Layer1, feed_dict=feed_dict_test))

        print("Optimization Finished!")

        # Testing
        t_test = time.time()
        test_loss, test_pred = sess.run([loss, pred], feed_dict=feed_dict_test)
        test_acc = masked_accuracy(test_pred, y_test, test_mask)
        print("Test set results:", "cost=", "{:.5f}".format(test_loss),
              "accuracy=", "{:.5f}".format(test_acc), "Test time=", "{:.5f}".format(time.time() - t_test))

        y_pred_res = sess.run(pred, feed_dict=feed_dict_test)

    file.write("GrAMME Supra Accuracy: " + str(max(val_accs)) + "\n")


if __name__ == '__main__':
    main(sys.argv)
