'''
Created on Dec 21, 2017

@author: purbasha
'''
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import keras

class AccLossPlotter(keras.callbacks.Callback):
    '''
    Plot training Accuracy and Loss values on a Matplotlib graph.
    The graph is updated by the 'on_epoch_end' event of the Keras Callback class\n",
    Arguments:
    graphs: list with some or all of ('acc', 'loss')
    save_graph: Save graph as an image on Keras Callback 'on_train_end' event
    '''
    def __init__(self, graphs=['acc', 'loss'], save_graph=False):
        self.graphs = graphs
        self.num_subplots = len(graphs)
        self.save_graph = save_graph

    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []
        self.epoch_count = 0
        plt.ion()
        plt.show()

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_count += 1,
        self.val_acc.append(logs.get('val_acc'))
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        epochs = [x for x in range(self.epoch_count)]
        count_subplots = 0
        if 'acc' in self.graphs:
            count_subplots += 1
            plt.subplot(self.num_subplots, 1, count_subplots)
            plt.title('Accuracy')
           #plt.axis([0,100,0,1])
            plt.plot(epochs, self.val_acc, color='r')
            plt.plot(epochs, self.acc, color='b')
            plt.ylabel('accuracy')
            red_patch = mpatches.Patch(color='red', label='Test')
            blue_patch = mpatches.Patch(color='blue', label='Train')
            plt.legend(handles=[red_patch, blue_patch], loc=4)
            if 'loss' in self.graphs:
                count_subplots += 1
                plt.subplot(self.num_subplots, 1, count_subplots)
                plt.title('Loss')
                #plt.axis([0,100,0,5])\n",
                plt.plot(epochs, self.val_loss, color='r')
                plt.plot(epochs, self.loss, color='b')
                plt.ylabel('loss')
                red_patch = mpatches.Patch(color='red', label='Test')
                blue_patch = mpatches.Patch(color='blue', label='Train')
    
from IPython.display import clear_output
class PlotLearning(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(20,8))
        clear_output(wait=True)
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        ax2.plot(self.x, self.acc, label="acc"),
        ax2.plot(self.x, self.val_acc, label="val_acc")
        ax2.legend()
        plt.show()

    "plot_learning = PlotLearning()\n"

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score, roc_curve, recall_score 
from itertools import cycle

def roc_pr_curves(y_test, model_test_scores, label, roc_flag, pr_flag):
    n_classes = 3
    plt.figure(figsize=(20, 8))
    # ROC CURVE\n",
    if roc_flag:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:,i], model_test_scores[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        plt.subplot(1, 2, 1)
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
            label='ROC curve of class {0} (area = {1:0.2f})'
          ''.format(i, roc_auc[i]))
        plt.title('Receiver Operating Characteristic')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label='ROC curve of class {0} (area = {1:0.2f})'
               ''.format(i, roc_auc[i]))
        plt.legend(loc='lower right')
        plt.plot([0,1], [0,1], linestyle='--', lw=2, color='r')
        plt.xlim([0,1])
        plt.ylim([0,1.01])
    #Precision Recall Curve\n",
    if pr_flag:
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test[:,i], model_test_scores[:,i], pos_label=True)
            average_precision[i] = average_precision_score(y_test[:,i], model_test_scores[:,i])
        plt.subplot(1, 2, 2)
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(precision[i], recall[i], color=color, lw=2,
                label='ROC curve of class {0} (area = {1:0.2f})'
               ''.format(i, average_precision[i]))
        plt.title('Precision Recall')
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.legend(loc='lower left')
        plt.xlim([0,1.01])
        plt.ylim([0,1.01])
    plt.show()
