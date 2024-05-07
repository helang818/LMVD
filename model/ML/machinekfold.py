import numpy as np
from re import T
import torch
import logging
from kfoldLoader import MyDataLoader 
from torch.utils.data import DataLoader
import math
from torch.optim.lr_scheduler import LambdaLR,MultiStepLR
from math import cos
from tqdm import tqdm
import torch.nn as nn
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearnex import patch_sklearn

classes = ['Normal','Depression']
totals = []
p_totals=[]
r_totals=[]
f1_totals=[]
tim = time.strftime('%m_%d__%H_%M', time.localtime())
filepath = 'the log file path'+str(tim)
savePath1 = "the model file path"+str(tim)

if not os.path.exists(filepath):
        os.makedirs(filepath)

logging.basicConfig(level=logging.NOTSET,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=filepath+'/'+'the model of ML.log',
                    filemode='w')


def plot_confusion_matrix(y_true, y_pred, labels_name, savename,title=None, thresh=0.6, axis_labels=None):
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()

    if title is not None:
        plt.title(title)

    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = classes
    plt.xticks(num_local, ['Normal','Depression'])
    plt.yticks(num_local, ['Normal','Depression'],rotation=90,va='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if cm[i][j] * 100 > 0:
                plt.text(j, i, format(cm[i][j] * 100 , '0.2f') + '%',
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")

    plt.savefig(savename, format='png')
    plt.clf()

def train(VideoPath, AudioPath, X_train,X_test,labelPath,numkfold):

    from sklearn.metrics import precision_score, recall_score, f1_score
    patch_sklearn()
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    trainSet = MyDataLoader(VideoPath, AudioPath,X_train,labelPath,  "train")
    trainLoader = DataLoader(trainSet, batch_size=2005, shuffle=True)
    devSet = MyDataLoader(VideoPath, AudioPath,X_test,labelPath,  "dev")
    devLoader = DataLoader(devSet, batch_size=1112, shuffle=False)
    for data1 in trainLoader:
        trainvideo,trainaudio,trainlable = data1
    for data2 in devLoader:
        devvideo,devaudio,devlable = data2
    print(trainvideo.shape,trainaudio.shape,trainlable.shape)
    X_train = torch.cat((trainvideo.view(trainvideo.shape[0],-1),trainaudio.view(trainaudio.shape[0],-1)),dim=1)
    X_train = X_train.numpy()
    X_lable = trainlable.numpy()
    Y_test =  torch.cat((devvideo.view(devvideo.shape[0],-1),devaudio.view(devaudio.shape[0],-1)),dim=1)
    Y_test =  Y_test.numpy()
    
    Y_lable = devlable.numpy()
    #model = SVC(kernel='rbf', C=1)
    #model = RandomForestClassifier(n_estimators=10, random_state=1234)
    #model = LogisticRegression()
    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X_train, X_lable)
    svc_predictions = model.predict(Y_test)
    accu = accuracy_score(Y_lable, svc_predictions) * 100
    
    p = precision_score(Y_lable, svc_predictions, average='weighted')
    r = recall_score(Y_lable, svc_predictions, average='weighted')
    f1score = f1_score(Y_lable, svc_predictions, average='weighted')
    
    p_totals.append(p)
    r_totals.append(r)
    f1_totals.append(f1score)

    totals.append(accu)
    print("train end")
    return Y_lable,svc_predictions


def count(string):
    dig = sum(1 for char in string if char.isdigit())
    return dig


if __name__ == '__main__':

    from sklearn.model_selection import KFold,StratifiedKFold
    seed = 2222
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    total_label=[]
    total_pre=[]
    from sklearn import metrics
        
    tcn = "the tcnfeature path"
    mdnAudioPath = "the audiofeature path"
    labelPath="the label file path"
    
       
    Y = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state = 42)
    X = os.listdir(tcn)
    X.sort(key = lambda x : int(x.split(".")[0]))
    X = np.array(X)
    
    for i in X:
        file_csv = pd.read_csv(os.path.join(labelPath,(str(i.split('.npy')[0])+"_Depression.csv")))
        bdi = int(file_csv.columns[0])
        Y.append(bdi)
    numkfold  = 0
    for train_index, test_index in kf.split(X,Y):
        X_train, X_test = X[train_index], X[test_index] 
        numkfold +=1
        logging.info('The {}  fold training set is:{}'.format(numkfold, X_train))
        logging.info('The {}  fold test set is:{}'.format(numkfold, X_test))
        label0,pre0 = train(tcn, mdnAudioPath,X_train,X_test,labelPath,numkfold)
        total_label.append(label0)
        total_pre.append(pre0)
    total_pre = np.concatenate(total_pre,axis = 0)
    
    total_label=np.concatenate(total_label,axis = 0)
    
    plot_confusion_matrix(total_label,total_pre,[0,1],savename=filepath+'/confusion_matrix.png')
    logging.info('The accuracy is：{}'.format(totals))
    logging.info('The average accuracy is：{}'.format(sum(totals)/len(totals)))
    logging.info('The precision is：{}'.format(p_totals))
    logging.info('The average precision is：{}'.format(sum(p_totals)/len(p_totals)))
    logging.info('The recall is：{}'.format(r_totals))
    logging.info('The average recall is：{}'.format(sum(r_totals)/len(r_totals)))
    logging.info('The f1 is：{}'.format(f1_totals))
    logging.info('The average f1 is：{}'.format(sum(f1_totals)/len(f1_totals)))
    print('acc:')
    print(totals)
    print(sum(totals)/len(totals))
    



