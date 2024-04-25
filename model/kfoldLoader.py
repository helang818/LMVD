

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

    def __init__(self, size, downsample):
        self.size = size
        self.downsample = downsample


    def __call__(self, frame_indices):
        downsample = max(min(int(self.vid_duration/self.size), self.downsample),1)

        end_index = min(self.begin_index + self.clip_duration, self.vid_duration)
        out = frame_indices[self.begin_index:end_index]
        while len(out) < self.clip_duration:
            for index in out:
                if len(out) >= self.clip_duration:
                    break
                out.append(index)
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


class MyDataLoader(udata.Dataset):
    def __init__(self, videoFileName, AudioFileName, Kfolds,labelPath,type) -> None:
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

        for file in Kfolds:
            file = str(file)
            id = file.split('.')[0]
            self.videoList.append(os.path.join(videoFileName, file))
            self.audioList.append(os.path.join(AudioFileName, file))
            
                
            file_csv = pd.read_csv(os.path.join(labelPath, file.replace(".npy", "_Depression.csv")))

            
            bdi = int(file_csv.columns[0])
            self.label.append(bdi)
            
    def __getitem__(self, index: int):

        videoData, audioData, label, = np.load(self.videoList[index]), np.load(self.audioList[index]), self.label[index]
        label = np.array(label)
        
        if self.temp is not None:
            videoData = self.temp(videoData)
        label = torch.from_numpy(label).type(torch.float)
        videoData = torch.from_numpy(videoData)
        audioData = torch.from_numpy(audioData)


        if audioData.shape[0] > normalAudioShape:
            audioData = audioData[:180,:]
        if videoData.shape[0] > normalVideoShape:
            videoData = videoData[:915,]
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

    def __len__(self) -> int:
        return len(self.videoList)

