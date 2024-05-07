# LMVD: A Large-Scale Multimodal Vlog Dataset for Depression Detection in the Wild

## Paper of our work

Here is the download link for our article "LMVD: A Large-Scale Multimodal Vlog Dataset for Depression Detection in the Wild":
- [Paper link](xxx)

## Introduction to Our Datasets
For audio features, the pre-trained VGGish model is adopted.

For visual features, FAU, facial landmarks, eye gaze, and head pose features are adopted.




## Requirements

- Python==3.8.0
- numpy==1.21.6
- torch==1.7.1
- torchvision==0.8.2
- pandas==2.0.3

## Using Our Datasets
We have made public the relevant code for our proposed model MDDformer, and the code involved in other comparative experiments may be made public later.

If you want to use our code, you may need to download the relevant data of our proposed dataset LMVD first.

For your convenience, we provide the following link:

- [Dataset download link for Baidu Netdisk](https://pan.baidu.com/s/1gviwLfbFcRSaARP5oT9yZQ?pwd=tvwa)

- [Dataset download link for figshare](https://figshare.com/articles/dataset/LMVD/25698351)

- [Dataset download link for IEEEDataPort](xxx)



## Running the Code:
In the code section, we have disclosed the relevant codes for four types of machine learning and four types of deep learning (add&concat). If you want to run the corresponding code, please refer to the following steps.

 - Machine Learning
  If you want to run machine learning code, please modify the relevant parts in machinekfold.py in \model\ML. Please note that the calling parts for the four machine learning methods are in lines 89 to 92.Please refer to the prompts in the code for other modifications.
  After you have modified all the code

`
python model/ML/machinekfold.py
`

 - Deep Learning
  For the code of deep learning, we will only take BiLSTM as an example,As with the previous operation, you only need to make simple modifications to the relevant parts in BILSTmfold.py in the model/BiLSTM/directory. If you want to switch between add and concat, you can modify the comments in BILSTMmodel.py.
 After you have modified all the code

`
python model/BiLSTM/BILSTMfold.py
`

  The operation of the other three methods is similar to the above.
