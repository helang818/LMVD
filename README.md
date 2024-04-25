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

- [Dataset download link for Google Drive](xxx)

- [Dataset download link for IEEEDataPort](xxx)

The verification code for Baidu Netdisk is```tvwa ```


## Running the Code:
If you want to run our code, please first modify lines 31 and 32 of \model\mddformer.py,Set the save path for your own log and model files.

Then, you should modify lines 282 to 284 and follow the prompts to modify the corresponding path.

After completing the above modifications, you can execute the following code to run our project:

`
python model/mddformer.py 
`
