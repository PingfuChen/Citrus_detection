# Citrus_detection
A kind of citrus detection method

#Install

git clone https://github.com/PingfuChen/Citrus_detection.git

cd Method

pip install -r requirements.txt

#Main work

This project mainly designs a Lightweight green citrus fruit detection method for practical environmental applications，which solves the problem of image blurring and degradation in the process of data acquisition in practical environmental applications and the real-time problem of citrus green fruit detection.

#Inference with detect.py

detect.py runs inference on a variety of sources, downloading models automatically from the latest YOLOv5 release and saving results to .runs/detect
python detect.py --sorce '' --data '' --weights ''

#Training

Use the following command to train,and you can train on your own dataset.
python train.py --data CitursVocdata.yaml --cfg our.yaml --weights '' --batch-size 16

#DataSet

Due to file transfer limitations, please contact the author for the datasets and models used in this project if necessary
E-mail:chenpingfu273@163.com
