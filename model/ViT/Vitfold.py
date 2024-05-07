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
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix
from Vitmodel import ViT
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
lr = 0.00001
epochSize = 200
warmupEpoch = 0
testRows = 1
schedule = 'consine'
classes = ['Normal','Depression']
ps = []
rs = []
f1s = []
totals = []

total_pre = []
total_label = []

tim = time.strftime('%m_%d__%H_%M', time.localtime())
filepath = 'the log file path'+str(tim)
savePath1 = "the model file path"+str(tim)

if not os.path.exists(filepath):
        os.makedirs(filepath)

logging.basicConfig(level=logging.NOTSET,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=filepath+'/'+'the log file name.log',
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



class AffectnetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset):
        print('initial balance sampler ...')

        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)

        expression_count = [0] * 63
        for idx in self.indices:
            label = dataset.label[idx]
            expression_count[int(label)] += 1

        self.weights = torch.zeros(self.num_samples)
        for idx in self.indices:
            label = dataset.label[idx]
            self.weights[idx] = 1. / expression_count[int(label)]

        print('initial balance sampler OK...')


    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))


    def __len__(self):
        return self.num_samples


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 0.5 * (cos(min((current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps),1) * math.pi) + 1)
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def train(VideoPath, AudioPath,X_train,X_test,labelPath,numkfold):
    mytop = 0
    topacc = 60
    top_p=0
    top_r=0
    top_f1=0
    top_pre=[]
    top_label=[]

    trainSet = MyDataLoader(VideoPath, AudioPath,X_train,labelPath,  "train")
    trainLoader = DataLoader(trainSet, batch_size=15, shuffle=True)
    devSet = MyDataLoader(VideoPath, AudioPath,X_test,labelPath,  "dev")
    devLoader = DataLoader(devSet, batch_size=4, shuffle=False)
    print("trainLoader finish", len(trainLoader), len(devLoader))

    if torch.cuda.is_available():
        model = ViT(spectra_size=128*2,patch_size=16,num_classes=2,dim=768,depth=4,heads=8,dim_mlp=128,channel=186,dim_head=8,dropout=0.5).cuda(1)

    lossFunc = nn.CrossEntropyLoss().cuda(1)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr,
                                    betas=(0.9,0.999),
                                    eps=1e-8,
                                    weight_decay=0,
                                    amsgrad=False
                                    )

    train_steps = len(trainLoader)*epochSize
    warmup_steps = len(trainLoader)*warmupEpoch
    target_steps = len(trainLoader)*epochSize
    
    if schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=train_steps)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=target_steps)

    logging.info('The {}  fold training begins！！'.format(numkfold))
    savePath=str(savePath1)+'/'+str(numkfold)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
        
    
    for epoch in range(1, epochSize):
        loop = tqdm(enumerate(trainLoader), total=len(trainLoader))
        traloss_one = 0
        correct = 0
        total = 0
        lable1 = []
        pre1 = []
        
        model.train()
        for batch_idx, (videoData, audioData, label) in loop:
            if torch.cuda.is_available():
                videoData, audioData, label = videoData.cuda(1), audioData.cuda(1), label.cuda(1)

            output = model(videoData, audioData)

            traLoss = lossFunc(output, label.long())
            traloss_one += traLoss
            optimizer.zero_grad()
            traLoss.backward()
            optimizer.step()
            scheduler.step()
            
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += predicted.eq(label.data).cpu().sum()

            loop.set_description(f'Train Epoch [{epoch}/{epochSize}]')
            loop.set_postfix(loss = traloss_one/(batch_idx+1))

        logging.info('EpochSize: {}, Train batch: {}, Loss:{}, Acc:{}%'.format(epoch, batch_idx+1, traloss_one/len(trainLoader), 100.0*correct/total))

        if epoch-warmupEpoch >=0 and epoch % testRows == 0:
            train_num = 0
            correct = 0
            total = 0
            dictt, labelDict = {},{}
            
            
            label2=[]
            pre2 = []
            
            model.eval()
            print("*******dev********")
            loop = tqdm(enumerate(devLoader), total=len(devLoader))
            with torch.no_grad():
                loss_one = 0
                for batch_idx, (videoData, audioData, label) in loop:
                    if torch.cuda.is_available():
                        videoData, audioData,label = videoData.cuda(1), audioData.cuda(1),label.cuda(1)
                    devOutput = model(videoData, audioData)
                    loss = lossFunc(devOutput, label.long())
                    loss_one += loss
                    train_num+=label.size(0)
                    
                    _, predicted = torch.max(devOutput.data, 1)
                    total += label.size(0)
                    correct += predicted.eq(label.data).cpu().sum()
                    
                    label2.append(label.data)
                    pre2.append(predicted)
                    
                    lable1 += label.data.tolist()
                    pre1 += predicted.tolist()
            
            acc = 100.0*correct/total
            lable1 = np.array(lable1)
            pre1 = np.array(pre1)

            p = precision_score(lable1, pre1, average='weighted')
            r = recall_score(lable1, pre1, average='weighted')
            f1score = f1_score(lable1, pre1, average='weighted')
            logging.info('precision:{}'.format(p))
            logging.info('recall:{}'.format(r))
            logging.info('f1:{}'.format(f1score))

            logging.debug('Dev epoch:{}, Loss:{}, Acc:{}%'.format(epoch,loss_one/len(devLoader), acc))
            loop.set_description(f'__Dev Epoch [{epoch}/{epochSize}]')
            loop.set_postfix(loss=loss)
            print('Dev epoch:{}, Loss:{},Acc:{}%'.format(epoch,loss_one/len(devLoader),acc))
            if acc> mytop:
                mytop = max(acc,mytop)
                top_p = p
                top_r = r
                top_f1 = f1score
                top_pre = pre2
                top_label = label2
                
            if acc > topacc:
                topacc = max(acc, topacc)
                checkpoint = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch, 'scheduler':scheduler.state_dict()}
                torch.save(checkpoint, savePath+'/'+"mdn+tcn"+'_'+str(epoch)+'_'+ str(acc)+'_'+ str(p)+'_'+str(r)+'_'+str(f1score)+'.pth')
                
    top_pre = torch.cat(top_pre,axis=0).cpu()
    top_label=torch.cat(top_label,axis=0).cpu()
    
    totals.append(mytop)
    ps.append(top_p)
    rs.append(top_r)
    f1s.append(top_f1)
    logging.info('topacc:'.format(mytop))
    logging.info('')
    
    print("train end")
    
    return top_label,top_pre

def count(string):
    dig = sum(1 for char in string if char.isdigit())
    return dig

if __name__ == '__main__':
    import random
    from sklearn.model_selection import KFold,StratifiedKFold
    seed = 2222
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
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
        total_label_0,total_pre_0 = train(tcn, mdnAudioPath,X_train,X_test,labelPath,numkfold)
        total_pre.append(total_pre_0)
        total_label.append(total_label_0)
        
    total_pre = torch.cat(total_pre,axis=0).cpu().numpy()
    total_label=torch.cat(total_label,axis=0).cpu().numpy()
    np.save(filepath+"/total_pre.npy",total_pre)
    np.save(filepath+"/total_label.npy",total_label)
    
    plot_confusion_matrix(total_label,total_pre,[0,1],savename=filepath+'/confusion_matrix.png')
    
    logging.info('The accuracy is：{}'.format(totals))
    logging.info('The average accuracy is：{}'.format(sum(totals)/len(totals)))
    logging.info('The precision is：{}'.format(ps))
    logging.info('The average precision is：{}'.format(sum(ps)/len(ps)))
    logging.info('The recall is：{}'.format(rs))
    logging.info('The average recall is：{}'.format(sum(rs)/len(rs)))
    logging.info('The f1 is：{}'.format(f1s))
    logging.info('The average f1 is：{}'.format(sum(f1s)/len(f1s)))
    print(totals)
    print(sum(totals)/len(totals))



