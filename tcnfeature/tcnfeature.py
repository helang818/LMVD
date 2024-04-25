import pandas as pd
import os
import numpy as np
import random
from pydub import AudioSegment
import math
import joblib
from sklearn.decomposition import PCA
                            


def validFrame(frames):
    
    for row in range(frames.shape[0]):
        if frames[row][4] == 1:
            validFrame = frames[row]
    for row in range(frames.shape[0]):
        if frames[row][4] == 0:
            frames[row][5:] = validFrame[5:]
        if frames[row][4] == 1:
            validFrame = frames[row]
    return frames




def chouzhen(_feature):
    flag = 0
    for i in range(0, len(_feature), 6):
        if flag == 0:
            feature = _feature[i]
            flag = 1
        else:
            feature = np.vstack((feature, _feature[i]))
    return feature



def split(data):
    

    _data = chouzhen(data[:5490, ])
    
    if _data.shape[0]<915:
        zeros = np.zeros([(915-_data.shape[0]),_data.shape[1]])
        _data = np.vstack((_data,zeros))
    return _data        
    

def getTCNVideoFeature(trainPath, targetPath):
    dirs = os.listdir(trainPath)
    for dir in dirs:
        files = os.listdir(os.path.join(trainPath, dir))
        for file in files:
             if file.split(".")[-1] == "csv":  
                file_csv = pd.read_csv(os.path.join(trainPath, dir, file))
                data = np.array(file_csv)
                try:
                    data = validFrame(data)
                except:
                    print('Video issues',file)
                    continue
                
                data = split(data)
                
                data = np.delete(data, [0,1,2,3,4], axis = 1)
                
                
                gaze = data[:, 0:6]
                gaze_zero = np.zeros_like(gaze)
                gaze = np.hstack((gaze, gaze_zero))
                pose = data[:, 288:294]
                features = data[:, 294:430]
                au = data[:, 447:465]
                
                au = np.delete(au, [5], axis = 1)
                au[:, [12,13]] = au[:, [13,12]]
                au[:, [13,14]] = au[:, [14,13]]               
                
                feature = au
                feature = np.hstack((feature, features, gaze, pose))

               
                try:

                    assert np.isnan(feature).sum() == 0, print(file)
                except:
                    print('There is a null value present：',file)
                
                np.save(os.path.join(targetPath,  file.split(".")[0]), feature)
                


if __name__ == "__main__":
    
    getTCNVideoFeature("Video feature path","save tcnfeature path")
    