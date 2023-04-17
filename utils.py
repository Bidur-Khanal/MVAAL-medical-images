import pdb
import os
import numpy as np
import nibabel as nib
import torch
import sys
import time
import logging
import logging.handlers
import pydensecrf.densecrf as dcrf

from pydensecrf.utils import compute_unary, create_pairwise_bilateral,\
         create_pairwise_gaussian, softmax_to_unary, unary_from_softmax

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score
from  sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table 

# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
term_width = 80 #(set the term_width to 80 directly)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def dice_coef(preds, targets, backprop=True):
    smooth = 1.0
    class_num = 2
    if backprop:
        for i in range(class_num):
            pred = preds[:,i,:,:]
            target = targets[:,i,:,:]
            intersection = (pred * target).sum()
            loss_ = 1 - ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
            if i == 0:
                loss = loss_
            else:
                loss = loss + loss_
        loss = loss/class_num
        return loss
    else:
        # Need to generalize

        targets = np.array(targets.cpu().numpy().argmax(1))

        if len(preds.shape) > 3:
            preds = np.array(preds.cpu().numpy()).argmax(1)
        for i in range(class_num):
            pred = (preds==i).astype(np.uint8)
            target= (targets==i).astype(np.uint8)
            intersection = (pred * target).sum()
            loss_ = 1 - ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
            if i == 0:
                loss = loss_
            else:
                loss = loss + loss_
        loss = loss/class_num
        return loss


def get_crf_img(inputs, outputs):
    for i in range(outputs.shape[0]):
        img = inputs[i]
        softmax_prob = outputs[i]
        unary = unary_from_softmax(softmax_prob)
        unary = np.ascontiguousarray(unary)
        d = dcrf.DenseCRF(img.shape[0] * img.shape[1], 2)
        d.setUnaryEnergy(unary)
        feats = create_pairwise_gaussian(sdims=(10,10), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        feats = create_pairwise_bilateral(sdims=(50,50), schan=(20,20,20),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(5)
        res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))
        if i == 0:
            crf = np.expand_dims(res,axis=0)
        else:
            res = np.expand_dims(res,axis=0)
            crf = np.concatenate((crf,res),axis=0)
    return crf


def erode_dilate(outputs, kernel_size=7):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    outputs = outputs.astype(np.uint8)
    for i in range(outputs.shape[0]):
        img = outputs[i]
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        outputs[i] = img
    return outputs


def post_process(args, inputs, outputs, input_path=None,
                 crf_flag=True, erode_dilate_flag=True,
                 save=True, overlap=True):
    inputs = (np.array(inputs.squeeze()).astype(np.float32)) * 255
    inputs = np.expand_dims(inputs, axis=3)
    inputs = np.concatenate((inputs,inputs,inputs), axis=3)
    outputs = np.array(outputs)

    # Conditional Random Field
    if crf_flag:
        outputs = get_crf_img(inputs, outputs)
    else:
        outputs = outputs.argmax(1)

    # Erosion and Dilation
    if erode_dilate_flag:
        outputs = erode_dilate(outputs, kernel_size=7)
    if save == False:
        return outputs

    outputs = outputs*255
    for i in range(outputs.shape[0]):
        path = input_path[i].split('/')
        output_folder = os.path.join(args.output_root, path[-2])
        try:
            os.mkdir(output_folder)
        except:
            pass
        output_path = os.path.join(output_folder, path[-1])
        if overlap:
            img = outputs[i]
            img = np.expand_dims(img, axis=2)
            zeros = np.zeros(img.shape)
            img = np.concatenate((zeros,zeros,img), axis=2)
            img = np.array(img).astype(np.float32)
            img = inputs[i] + img
            if img.max() > 0:
                img = (img/img.max())*255
            else:
                img = (img/1) * 255
            cv2.imwrite(output_path, img)
        else:
            img = outputs[i]
            cv2.imwrite(output_path, img)
    return None


class Checkpoint:
    def __init__(self, model, optimizer=None, epoch=0, best_score=1):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.best_score = best_score

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state"])
        self.epoch = checkpoint["epoch"]
        self.best_score = checkpoint["best_score"]
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            for state in self.optimizer.state.values():
                  for k, v in state.items():
                           if torch.is_tensor(v):
                                    state[k] = v.cuda()

    def save(self, path):
        state_dict = self.model.module.state_dict()
        torch.save({"model_state": state_dict,
                    "optimizer_state": self.optimizer.state_dict(),
                    "epoch": self.epoch,
                    "best_score": self.best_score}, path)


def progress_bar(current, total, msg=None):
    ''' Source Code from 'kuangliu/pytorch-cifar'
        (https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py)
    '''
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    ''' Source Code from 'kuangliu/pytorch-cifar'
        (https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py)
    '''
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def get_logger(level="DEBUG", file_level="DEBUG"):
    logger = logging.getLogger(None)
    logger.setLevel(level)
    fomatter = logging.Formatter(
            '%(asctime)s  [%(levelname)s]  %(message)s  (%(filename)s:  %(lineno)s)')
    fileHandler = logging.handlers.TimedRotatingFileHandler(
            'result.log', when='d', encoding='utf-8')
    fileHandler.setLevel(file_level)
    fileHandler.setFormatter(fomatter)
    logger.addHandler(fileHandler)
    return logger


#### function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    #transpose the matrix to make x-axis True Class and Y-axis Predicted Class
    cm= np.transpose(cm)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

   
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[0]),
           yticks=np.arange(cm.shape[1]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           
           #here we are not printing the title
           #title=title,
           xlabel='True label',
           ylabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig,ax



# compute the AUC per class 
# implementation adapted from 
# https://stackoverflow.com/questions/39685740/calculate-sklearn-roc-auc-score-for-multi-class
def class_report(y_true, y_pred, y_score=None, average='micro', sklearn_cls_report = True):

    if isinstance(y_true, list):
        y_true = np.array(y_true)

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return

    if sklearn_cls_report:
       
        class_report_df = pd.DataFrame(classification_report(y_true= y_true, y_pred = y_pred, output_dict=True)).transpose()
    else:

        lb = LabelBinarizer()

        if len(y_true.shape) == 1:
            lb.fit(y_true)

        #Value counts of predictions
        labels, cnt = np.unique(
            y_pred,
            return_counts=True)
        n_classes = len(labels)
        pred_cnt = pd.Series(cnt, index=labels)

        metrics_summary = precision_recall_fscore_support(
                y_true=y_true,
                y_pred=y_pred,
                labels=labels)

        avg = list(precision_recall_fscore_support(
                y_true=y_true, 
                y_pred=y_pred,
                average='weighted'))

        metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
        class_report_df = pd.DataFrame(
            list(metrics_summary),
            index=metrics_sum_index,
            columns=labels)

        support = class_report_df.loc['support']
        total = support.sum() 
        class_report_df['avg / total'] = avg[:-1] + [total]

        class_report_df = class_report_df.T
        class_report_df['pred'] = pred_cnt
        class_report_df['pred'].iloc[-1] = total


       
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    # Create the table and add to the axis
    table = ax.table(cellText=class_report_df.values, colLabels=class_report_df.columns, loc='center')

    # Set the font size of the table
    table.set_fontsize(14)

    # Remove the axis and add a title
    ax.axis('off')
    ax.set_title('Sample Table')

    return fig


    #return class_report_df


def computeAUROC(dataGT, dataPRED, classCount):
    # Computes area under ROC curve 
    # dataGT: ground truth data
    # dataPRED: predicted data

    if classCount == 1:
        outAUROC = roc_auc_score(dataGT, dataPRED)
    else:
        outAUROC = []
        for i in range(classCount):
            try:
                outAUROC.append(roc_auc_score(dataGT[:, i], dataPRED[:, i]))
            except ValueError:
                outAUROC.append(0)
    
    return outAUROC


def get_acc(true_labels, predicted_labels):
    """Args: 
        true_labels: given groundtruth labels
        predicted_labels: predicted labels from the model
    Returns:
        overall_acc: overll accuracy 
    """

    overall_acc = accuracy_score(true_labels, predicted_labels)
    return overall_acc


def get_f1_score(true_labels, predicted_labels, average = "binary"):
    """Args: 
        true_labels: given groundtruth labels
        predicted_labels: predicted labels from the model
    Returns:
       f1score: F1 score of data
    """
    if average == "binary":
        f1score = f1_score(true_labels, predicted_labels)
    else:
        f1score = f1_score(true_labels, predicted_labels, average = "weighted")
    return f1score

