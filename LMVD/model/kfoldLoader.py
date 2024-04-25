

import torch.utils.data as udata
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import torch
from pydub import AudioSegment
import random
import logging
from torchvision import transforms
from PIL import Image

normalVideoShape = 915
normalAudioShape = 186

class temporalMask():
    def __init__(self, drop_ratio):
        self.ratio = drop_ratio
    def __call__(self, frame_indices):
        frame_len = frame_indices.shape[0]
        sample_len = int(self.ratio*frame_len)
        sample_list = random.sample([i for i in range(0, frame_len)], sample_len)
        frame_indices[sample_list,:]=0
        return frame_indices

class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size, downsample):
        self.size = size
        self.downsample = downsample


    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        # vid_duration  = frame_indices.shape[0]
        downsample = max(min(int(self.vid_duration/self.size), self.downsample),1)
        # clip_duration = self.size * downsample

        # rand_end = max(0, vid_duration - clip_duration-1)
        # begin_index = random.randint(0, rand_end)
        end_index = min(self.begin_index + self.clip_duration, self.vid_duration)
        # print(begin_index, end_index)
        out = frame_indices[self.begin_index:end_index]
        # random_out = [random.randint(0,downsample-1) for i in range(0,self.size)]
        # print(random_out)
        # print(out.shape)
        while len(out) < self.clip_duration:
            for index in out:
                if len(out) >= self.clip_duration:
                    break
                # out = np.vstack((out,index))
                out.append(index)
            # print("new_shape: ",out.shape)
        # selected_frames = [out[i] for i in range(0, self.clip_duration, downsample)]
        selected_frames = [i for i in range(self.begin_index, self.clip_duration+self.begin_index, downsample)]
        return selected_frames
    
    def randomize_parameters(self, frame_indices):
        self.vid_duration  = len(frame_indices)
        downsample = max(min(int(self.vid_duration/self.size), self.downsample),1)
        self.clip_duration = self.size * downsample

        rand_end = max(0, self.vid_duration - self.clip_duration-1)
        self.begin_index = random.randint(0, rand_end)
    
class AffectnetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset):
        # print('initial balance sampler ...')

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

        # print('initial balance sampler OK...')

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

def chouzhen(_feature):
    flag = 0
    for i in range(0, len(_feature), 6):
        if flag == 0:
            feature = _feature[i]
            flag = 1
        else:
            feature = np.vstack((feature, _feature[i]))
    return feature




'''
class MyDataLoader(udata.Dataset):
    def __init__(self, videoFileName, videoFileNameMDN, AudioFileName, Kfolds,type) -> None:
        super().__init__()
        labelPath = "/raid/junnan/zj/Depression_Lable/"
        if type == "train":
            self.temp = temporalMask(0.25)
        else :
            self.temp = None
        
        self.transform = TemporalRandomCrop(size=16, downsample=4)
        self.videoList = []
        self.audioList = []
        self.videoListMDN = []
        self.label = []
        self.type = type

        # TCN + MDN
        for file in Kfolds:
            file = str(file)
            id = file.split('.')[0]
            self.videoList.append(os.path.join(videoFileName, file))
            self.audioList.append(os.path.join(AudioFileName, file))
            
                
            self.videoListMDN.append(os.path.join(videoFileNameMDN, id, id+"_aligned"))
            file_csv = pd.read_csv(os.path.join(labelPath, file.replace(".npy", "_Depression.csv")))

            
            bdi = int(file_csv.columns[0])
            self.label.append(bdi)
            # self.gender.append(genderDict[file.replace("npy","mp4")])
'''
class MyDataLoader(udata.Dataset):
    def __init__(self, videoFileName, videoFileNameMDN, AudioFileName, Kfolds,labelPath,type) -> None:
        super().__init__()
        if type == "train":
            self.temp = temporalMask(0.25)
        else :
            self.temp = None
        
        self.transform = TemporalRandomCrop(size=16, downsample=4)
        self.videoList = []
        self.audioList = []
        self.label = []
        self.type = type

        # TCN + MDN
        for file in Kfolds:
            file = str(file)
            id = file.split('.')[0]
            self.videoList.append(os.path.join(videoFileName, file))
            self.audioList.append(os.path.join(AudioFileName, file))
            
                
            file_csv = pd.read_csv(os.path.join(labelPath, file.replace(".npy", "_Depression.csv")))

            
            bdi = int(file_csv.columns[0])
            self.label.append(bdi)
            # self.gender.append(genderDict[file.replace("npy","mp4")])
            
    def __getitem__(self, index: int):

        videoData, audioData, label, = np.load(self.videoList[index]), np.load(self.audioList[index]), self.label[index]
        label = np.array(label)
        
        if self.temp is not None:
            videoData = self.temp(videoData)
        label = torch.from_numpy(label).type(torch.float)
        videoData = torch.from_numpy(videoData)
        audioData = torch.from_numpy(audioData)
        # videoDataMDN = torch.from_numpy(videoDataMDN)


        if audioData.shape[0] > normalAudioShape:
            audioData = audioData[:180,:]
        if videoData.shape[0] > normalVideoShape:
            videoData = videoData[:915,]
         # 表达式为false时，触发assert
        assert videoData.shape[0] <= normalVideoShape
        assert audioData.shape[0] <= normalAudioShape
        assert videoData.shape[0] > 0
        assert audioData.shape[0] > 0

        if videoData.shape[0] < normalVideoShape:
            zeroPadVideo = nn.ZeroPad2d(padding=(0,0,0,normalVideoShape-videoData.shape[0]))
            videoData = zeroPadVideo(videoData)
        if audioData.shape[0] < normalAudioShape:
            zeroPadAudio = nn.ZeroPad2d(padding=(0,0,0,normalAudioShape-audioData.shape[0]))
            audioData = zeroPadAudio(audioData)
        
        videoData = videoData.type(torch.float)
        audioData = audioData.type(torch.float)
        
        if self.type == "train":
            return videoData, audioData, label
        if self.type == "dev":
            return videoData, audioData, label
    '''
    def __getitem__(self, index: int):

        files = os.listdir(self.videoListMDN[index])
        files.sort(key = lambda x : int(x[:-4].split("_")[-1])) 
        imgList = []
        imgNamePath = []
        for i, img_i in enumerate(files):
            if i == 5490:
                break
            # img_i = Image.open(os.path.join(self.videoList[index], img_i))
            # imgList.append(img_i)
            imgNamePath.append(os.path.join(self.videoListMDN[index], img_i))
        # toTensor = transforms.ToTensor() 
        self.transform.randomize_parameters(imgNamePath)   
        
        videoPathList = self.transform(imgNamePath)
        video = []
        for path in videoPathList:
            img_i = Image.open(imgNamePath[path])
            video.append(img_i)
        Transform = transforms.Compose([
                    transforms.Resize((112,112)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
        clip = [Transform(img_arr) for img_arr in video]
        videoDataMDN = torch.stack(clip, 0).permute(1, 0, 2, 3)

        videoData, audioData, label, = np.load(self.videoList[index]), np.load(self.audioList[index]), self.label[index]
        label = np.array(label)
        
        if self.temp is not None:
            videoData = self.temp(videoData)
            videoDataMDN = self.temp(videoDataMDN)
        label = torch.from_numpy(label).type(torch.float)
        videoData = torch.from_numpy(videoData)
        audioData = torch.from_numpy(audioData)
        # videoDataMDN = torch.from_numpy(videoDataMDN)


        if audioData.shape[0] > normalAudioShape:
            audioData = audioData[:180,:]
        if videoData.shape[0] > normalVideoShape:
            videoData = videoData[:915,]
         # 表达式为false时，触发assert
        assert videoData.shape[0] <= normalVideoShape
        assert audioData.shape[0] <= normalAudioShape
        assert videoData.shape[0] > 0
        assert audioData.shape[0] > 0

        if videoData.shape[0] < normalVideoShape:
            zeroPadVideo = nn.ZeroPad2d(padding=(0,0,0,normalVideoShape-videoData.shape[0]))
            videoData = zeroPadVideo(videoData)
        if audioData.shape[0] < normalAudioShape:
            zeroPadAudio = nn.ZeroPad2d(padding=(0,0,0,normalAudioShape-audioData.shape[0]))
            audioData = zeroPadAudio(audioData)
        
        videoData = videoData.type(torch.float)
        audioData = audioData.type(torch.float)
        videoDataMDN = videoDataMDN.type(torch.float)
        
        if self.type == "train":
            return videoData, audioData, videoDataMDN, label
        if self.type == "dev":
            return videoData, audioData, videoDataMDN, label, self.videoListMDN[index]
        '''
  
    def __len__(self) -> int:
        return len(self.videoList)

