import numpy as np
import tensorflow as tf
from absl import app, flags
import argparse
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Activation
from tensorflow.keras.models import Sequential

from attacks.projected_gradient_descent import projected_gradient_descent
from attacks.fast_gradient_method import fast_gradient_method
from attacks.cw import cw
from attacks.momentum_iterative_method import momentum_iterative_method

from crowd.crowd_layers import CrowdsClassification, MaskedMultiCrossEntropy, MaskedMultiCrossEntropy_cl
from crowd.agg import agg, adv_agg
from utils import load_data
from tqdm import tqdm
import os
import random

FLAGS = flags.FLAGS

parser = argparse.ArgumentParser(description='A-LFC method', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', type=str, default='music',
                    choices=['music','label_me','sentiment'],
                    help="What dataset to use.")
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--batch_size', type=int, default=64,
                    help='reduce this if your GPU runs out of memory')
parser.add_argument('--nb_epoch', type=int, default=200,
                    help='increase this if you think the model does not overfit')
parser.add_argument('--adv_training_eps', type=float, default=8/255,
                    help='epsilon during adversarial training')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Harmonic parameters of normal samples and adversarial samples')
parser.add_argument('--fn', type=str, default='gt',
                    choices=['gt', 'mv', 'ds', 'agg','cl','adv_training_agg'],
                    help="Crowdsourcing Function.")
parser.add_argument('--output', type=bool, default=False,
                    help='output test accuracy') 
parser.add_argument('--blackBox', type=bool, default=False,
                    help='blackBox')               
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

# Configuration parameters
BATCH_SIZE = args.batch_size
N_EPOCHS = args.nb_epoch

data_train_vgg16, labels_train_bin, labels_train_mv_bin, labels_train_ds_bin, data_test_vgg16, labels_test_bin, answers_bin_missings, answers, labels_test, N_CLASSES, N_ANNOT = load_data(args.dataset)

class Net(Model):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = Flatten()
        self.dense1 = Dense(128, input_shape=(124,), activation='relu')
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(N_CLASSES)
        self.ac = Activation("softmax")
        
    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return self.ac(x)

class genNet(Model):
    def __init__(self):
        super(genNet, self).__init__()
        self.flatten = Flatten()
        self.dense1 = Dense(32, input_shape=(124,), activation='relu')
        self.dropout = Dropout(0.3)
        self.dense2 = Dense(N_CLASSES)
        self.ac = Activation("softmax")
        
    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return self.ac(x)

def main(_):
    if args.blackBox:
        genModel = Sequential()
        genModel.add(genNet())
        genModel.compile(optimizer='adam', loss='categorical_crossentropy')
        genModel.fit(data_train_vgg16, labels_train_mv_bin, epochs=N_EPOCHS, shuffle=True, batch_size=BATCH_SIZE,
                  verbose=0)
    model = Sequential()
    model.add(Net())

    # Metrics to track the different accuracies.
    test_acc_clean = tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None)
    test_acc_fgsm = tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None)
    test_acc_pgd = tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None)
    test_acc_cw = tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None)
    test_acc_bim = tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None)
    test_acc_madry = tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None)
    test_acc_mim = tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None)
    test_acc_adv_mim = tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None)
    
    ################################### gt
    if FLAGS.fn == "gt":
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.fit(data_train_vgg16, labels_train_bin, epochs=N_EPOCHS, shuffle=True, batch_size=BATCH_SIZE, verbose=0)
    ################################### mv
    elif FLAGS.fn == "mv":
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.fit(data_train_vgg16, labels_train_mv_bin, epochs=N_EPOCHS, shuffle=True, batch_size=BATCH_SIZE,
                  verbose=0)
    ################################### ds
    elif FLAGS.fn == "ds":
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.fit(data_train_vgg16, labels_train_ds_bin, epochs=N_EPOCHS, shuffle=True, batch_size=BATCH_SIZE,
                  verbose=0)
    ################################### Aggnet
    elif FLAGS.fn == "agg":
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        result_aggnet = []
        model.fit(data_train_vgg16, labels_train_mv_bin, epochs=N_EPOCHS, shuffle=True, batch_size=BATCH_SIZE, verbose=0)
        model, _ = agg(model, data_train_vgg16, answers, N_CLASSES).run(200) 
    ################################### crowd layer
    elif FLAGS.fn == "cl":
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.add(CrowdsClassification(N_CLASSES, N_ANNOT, conn_type="MW"))
        loss = MaskedMultiCrossEntropy_cl().loss
        model.compile(optimizer='adam', loss=loss)
        model.fit(data_train_vgg16, answers_bin_missings, epochs=N_EPOCHS, shuffle=True, batch_size=BATCH_SIZE,
                  verbose=0)
        model.pop()
    ################################## adversarial training based on Aggnet
    elif FLAGS.fn == "adv_training_agg":
        def my_loss(y_true, y_pred):
            y_true_split = tf.split(y_true,2,0)
            y_pred_split = tf.split(y_pred,2,0)

            loss_nor = tf.keras.losses.categorical_crossentropy(y_true_split[0], y_pred_split[0])
            loss_adv = tf.keras.losses.categorical_crossentropy(y_true_split[1], y_pred_split[1])
            loss = (1 - args.alpha) * loss_nor + args.alpha * loss_adv 
            return loss

        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.fit(data_train_vgg16, labels_train_mv_bin, epochs=N_EPOCHS, shuffle=True, batch_size=BATCH_SIZE, verbose=0)
        model, _  = agg(model, data_train_vgg16, answers, N_CLASSES).run(10)
        adv_examples = projected_gradient_descent(model, data_train_vgg16, args.adv_training_eps, args.adv_training_eps / 4, 10, np.inf, y=labels_train_mv_bin, loss_fn=tf.keras.losses.categorical_crossentropy, dataset=args.dataset)
        model.compile(optimizer='adam', loss=my_loss)
        for i in tqdm(range(20)):
            model, ground_truth_est, _ = adv_agg(model, data_train_vgg16, adv_examples, answers, N_CLASSES, alpha = args.alpha).run(10)
            adv_examples = projected_gradient_descent(model, data_train_vgg16, args.adv_training_eps, args.adv_training_eps / 4, 10, np.inf, y=ground_truth_est, loss_fn=tf.keras.losses.categorical_crossentropy, dataset=args.dataset)

    if args.dataset != "sentiment":
        epsilons = [0.001, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    else:
        epsilons = [0.001, 0.002, 0.0025, 0.0030, 0.0040, 0.0050, 0.0075, 0.01]

    output_results = []
    for epsilon in epsilons:
        test_acc_clean.reset_states()
        test_acc_fgsm.reset_states()
        test_acc_pgd.reset_states()
        test_acc_cw.reset_states()
        test_acc_mim.reset_states()

        # Evaluate on clean and adversarial data
        y_pred = model(data_test_vgg16)
        test_acc_clean.update_state(labels_test_bin, y_pred)

        if args.blackBox:
            x_fgm = fast_gradient_method(genModel, data_test_vgg16, epsilon, np.inf, y=labels_test_bin, loss_fn=tf.keras.losses.categorical_crossentropy, dataset=args.dataset)
        else:
            x_fgm = fast_gradient_method(model, data_test_vgg16, epsilon, np.inf, y=labels_test_bin, loss_fn=tf.keras.losses.categorical_crossentropy, dataset=args.dataset)
        y_pred_fgm = model(x_fgm)
        test_acc_fgsm.update_state(labels_test_bin, y_pred_fgm)

        if args.blackBox:
            x_pgd = projected_gradient_descent(genModel, data_test_vgg16, epsilon, epsilon/4, 10, np.inf, y=labels_test_bin, loss_fn=tf.keras.losses.categorical_crossentropy, dataset=args.dataset)
        else:
            x_pgd = projected_gradient_descent(model, data_test_vgg16, epsilon, epsilon/4, 10, np.inf, y=labels_test_bin, loss_fn=tf.keras.losses.categorical_crossentropy, dataset=args.dataset)
        y_pred_pgd = model(x_pgd)
        test_acc_pgd.update_state(labels_test_bin, y_pred_pgd)

        if args.blackBox:
            x_cw = cw(genModel, data_test_vgg16, epsilon, epsilon/4, 10, np.inf, y=labels_test_bin, loss_fn=tf.keras.losses.categorical_crossentropy, dataset=args.dataset)
        else:
            x_cw = cw(model, data_test_vgg16, epsilon, epsilon/4, 10, np.inf, y=labels_test_bin, loss_fn=tf.keras.losses.categorical_crossentropy, dataset=args.dataset)            
        y_pred_cw = model(x_cw)
        test_acc_cw.update_state(labels_test_bin, y_pred_cw)
        
        if args.blackBox:
            x_mim = momentum_iterative_method(genModel, data_test_vgg16, epsilon, epsilon/4, 10, np.inf, y=labels_test_bin, loss_fn=tf.keras.losses.categorical_crossentropy, dataset=args.dataset)
        else:
            x_mim = momentum_iterative_method(model, data_test_vgg16, epsilon, epsilon/4, 10, np.inf, y=labels_test_bin, loss_fn=tf.keras.losses.categorical_crossentropy, dataset=args.dataset)            
        y_pred_mim = model(x_mim)
        test_acc_mim.update_state(labels_test_bin, y_pred_mim)
        
        output_results.append(test_acc_clean.result().numpy() * 100)
        output_results.append(test_acc_fgsm.result().numpy() * 100)
        output_results.append(test_acc_pgd.result().numpy() * 100)
        output_results.append(test_acc_cw.result().numpy() * 100)
        output_results.append(test_acc_mim.result().numpy() * 100)
    
    output_results = np.array(output_results).reshape((-1,5))
    if args.output:
        filepath = "./output/"
        if args.fn != "adv_training_agg":
            if args.blackBox:
                filename = str(args.fn) + "_" + str(args.dataset) + "_black_" + str(args.seed) + ".txt"
            else:
                filename = str(args.fn) + "_" + str(args.dataset) + "_" + str(args.seed) + ".txt"
        else:
            if args.blackBox:
                filename = "3_" + str(args.fn) + "_" + str(args.dataset) + "_" + str(args.alpha) + "_black_" + str(args.seed) + ".txt"
            else:
                filename = "3_" + str(args.fn) + "_" + str(args.dataset) + "_" + str(args.alpha) + "_" + str(args.seed) + ".txt"
        np.savetxt(filepath + filename, output_results)
    print(str(args.fn) + "_" + str(args.dataset) + "_" + str(args.seed) + " OK!")

if __name__ == "__main__":
    flags.DEFINE_string("fn", "agg", "Crowdsourcing Function.")
    flags.DEFINE_string("dataset", "music", "Dateset.")
    flags.DEFINE_integer("batch_size", 64, "batch_size.")
    flags.DEFINE_integer("nb_epoch", 200, "nb_epoch.")
    flags.DEFINE_integer("seed", 0, "random seed.")
    flags.DEFINE_float("alpha", 0.5, "Harmonic parameters.")
    flags.DEFINE_float("adv_training_eps", 8/255, "epsilon during adversarial training.")
    flags.DEFINE_boolean("output", False, "output test accuracy.")
    flags.DEFINE_boolean("blackBox", False, "blackBox.")

    app.run(main)