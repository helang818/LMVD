import numpy as np
from re import T
#from MDNTCNModel import *
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
from tcnmodel import Net
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
lr = 0.00001
epochSize = 300
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
filepath = '/home/junnanzhao/test_project/log/'+str(tim)
savePath1 = "/home/junnanzhao/test_project/model/"+str(tim)

if not os.path.exists(filepath):
        os.makedirs(filepath)
        
#设置日志内容到/temp/myapp.log文件
logging.basicConfig(level=logging.NOTSET,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=filepath+'/'+'newtcn_ffn2_total_norm_weighted.log',
                    filemode='w')


def plot_confusion_matrix(y_true, y_pred, labels_name, savename,title=None, thresh=0.6, axis_labels=None):
# 利用sklearn中的函数生成混淆矩阵并归一化
    #plt.figure(figsize=(10, 8), dpi=100)
    #plt.rcParams.update({ 'font.size': 16 })
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵 
    #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
# 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()  # 绘制图例

# 图像标题
    if title is not None:
        plt.title(title)
# 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = classes
    plt.xticks(num_local, ['Normal','Depression'])  # 将标签印在x轴坐标上
    plt.yticks(num_local, ['Normal','Depression'],rotation=90,va='center')  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if cm[i][j] * 100 > 0:
                plt.text(j, i, format(cm[i][j] * 100 , '0.2f') + '%',
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
# 显示
    #plt.savefig("/home/junnanzhao/test_project/SVM/confusion_matrix.png", format='png')
    
    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.clf()



class AffectnetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset):
        print('initial balance sampler ...')

        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)

        expression_count = [0] * 63 # 共8题，每题0-3分
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

def train(trainVideoPath, trainVideoPathMDN, devVideoPath, devVideoPathMDN, trainAudioPath, devAudioPath,X_train,X_test,labelPath,numkfold):
    mytop = 0
    topacc = 60
    top_p=0
    top_r=0
    top_f1=0
    top_pre=[]
    top_label=[]

    trainSet = MyDataLoader(trainVideoPath, trainVideoPathMDN, trainAudioPath,X_train,labelPath,  "train")
    # sampler = AffectnetSampler(trainSet)
    # trainLoader = DataLoader(trainSet, batch_size=20, shuffle=False, sampler=sampler)
    trainLoader = DataLoader(trainSet, batch_size=4, shuffle=True)
    devSet = MyDataLoader(devVideoPath, devVideoPathMDN, devAudioPath,X_test,labelPath,  "dev")
    devLoader = DataLoader(devSet, batch_size=4, shuffle=False) 
    # torch.Size([999, 915, 171]) torch.Size([999, 186, 128]) torch.Size([999])
    print("trainLoader finish", len(trainLoader), len(devLoader))

    # 加载模型到GPU
    if torch.cuda.is_available():
        model = Net().cuda(1)

    # 均方损失函数
    #lossFunc = nn.MSELoss().cuda(1)
    # lossFunc = nn.SmoothL1Loss()
    lossFunc = nn.CrossEntropyLoss().cuda(1)

    # 优化器
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
    # checkpoint = "/home/ywz/DANN_myself/model/test_30_6.761648633075345-5.50971368660134.pth"
    # if checkpoint is not None:
    #     checkpoint = torch.load(checkpoint)
    #     model.load_state_dict(checkpoint['net'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
    
    #scheduler = MultiStepLR(optimizer, milestones=[10,50,100,150], gamma=0.9)
    logging.info('第 {} 折训练开始！！'.format(numkfold))
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
            # 梯度归零
            #print(videoData.shape, audioData.shape, videoDataMDN.shape, label)
            output = model(videoData, audioData)
            #print(output)
            #print(output.shape)
            traLoss = lossFunc(output, label.long()) # 算损失
            traloss_one += traLoss
            optimizer.zero_grad()
            traLoss.backward() # 反向传播
            optimizer.step() # 参数更新
            scheduler.step()
            
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += predicted.eq(label.data).cpu().sum()
            
            # logging.warning('output:{}, label:{}'.format(output, label.float()))
            loop.set_description(f'Train Epoch [{epoch}/{epochSize}]')
            loop.set_postfix(loss = traloss_one/(batch_idx+1))
            #print('Train batch: {}, Loss{}'.format(batch_idx+1, traLoss))
        
        logging.info('EpochSize: {}, Train batch: {}, Loss:{}, Acc:{}%'.format(epoch, batch_idx+1, traloss_one/len(trainLoader), 100.0*correct/total))
        
        # 验证 每testRows验证一次
        if epoch-warmupEpoch >=0 and epoch % testRows == 0:
            train_num = 0
            correct = 0
            total = 0
            dictt, labelDict = {},{}
            
            
            label2=[]
            pre2 = []
            
            model.eval() # 冻结参数
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
                    #print(lable1)
                    #print(pre1)
                    
                    
            #pre_0 = torch.cat(pre2,axis=0).cpu().numpy()
            #label_0=torch.cat(lable2,axis=0).cpu().numpy()
            
            
            acc = 100.0*correct/total
            lable1 = np.array(lable1)
            pre1 = np.array(pre1)

            p = precision_score(lable1, pre1, average='weighted')
            r = recall_score(lable1, pre1, average='weighted')
            f1score = f1_score(lable1, pre1, average='weighted')
            logging.info('precision:{}'.format(p))
            logging.info('recall:{}'.format(r))
            logging.info('f1:{}'.format(f1score))
            # logging.debug('output:{}, label:{}, loss:{}'.format(devOutput, label, loss))
                    
            # loop.set_postfix(loss=loss_one/(batch_idx+1))


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
    
    '''    
    tcn = "/raid/junnan/zj/tcnfeature"
    mdntrainVideoPath = "/raid/junnan/zj/images"
    mdndevVideoPath = "/raid/junnan/zj/images"
    mdnAudioPath = "/raid/junnan/zj/audiofeature"
    '''
    tcn = "/raid/junnan/zj/tcnfeature_total"
    mdntrainVideoPath = "/raid/junnan/zj/images"
    mdndevVideoPath = "/raid/junnan/zj/images"
    mdnAudioPath = "/raid/junnan/zj/audiofeature_total"
    labelPath="/raid/junnan/zj/total_lable/"
    
    Y = []
    #kf = StratifiedKFold(n_splits=10, random_state=None) # 10折
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state = 42)
    X = os.listdir(tcn)
    X.sort(key = lambda x : int(x.split(".")[0]))
    X = np.array(X)
    
    for i in X:
        file_csv = pd.read_csv(os.path.join(labelPath,(str(i.split('.npy')[0])+"_Depression.csv")))
        bdi = int(file_csv.columns[0])
        Y.append(bdi)
        
    #print(type(X))
    #print(type(X[0]))
    numkfold  = 0
    #显示具体划分情况
    for train_index, test_index in kf.split(X,Y):
        #print("Train:", train_index, "Validation:",test_index)
        X_train, X_test = X[train_index], X[test_index] 
        numkfold +=1
        logging.info('第 {} 折 训练集为:{}'.format(numkfold, X_train))
        logging.info('第 {} 折 测试集为:{}'.format(numkfold, X_test))
        #print(i)
        #print(X_train)
        #print(X_test)
        total_label_0,total_pre_0 = train(tcn, mdntrainVideoPath, tcn, mdndevVideoPath, mdnAudioPath, mdnAudioPath,X_train,X_test,labelPath,numkfold)
        total_pre.append(total_pre_0)
        total_label.append(total_label_0)
        
    total_pre = torch.cat(total_pre,axis=0).cpu().numpy()
    total_label=torch.cat(total_label,axis=0).cpu().numpy()
    np.save(filepath+"/total_pre.npy",total_pre)
    np.save(filepath+"/total_label.npy",total_label)
    
    plot_confusion_matrix(total_label,total_pre,[0,1],savename=filepath+'/confusion_matrix.png')
    
    logging.info('准确率为：{}'.format(totals))
    logging.info('平均准确率为：{}'.format(sum(totals)/len(totals)))
    logging.info('precision：{}'.format(ps))
    logging.info('平均precision为：{}'.format(sum(ps)/len(ps)))
    logging.info('recall为：{}'.format(rs))
    logging.info('平均recall为：{}'.format(sum(rs)/len(rs)))
    logging.info('f1为：{}'.format(f1s))
    logging.info('平均f1为：{}'.format(sum(f1s)/len(f1s)))
    print(totals)
    print(sum(totals)/len(totals))    



