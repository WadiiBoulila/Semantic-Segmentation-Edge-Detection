# A Semantic Segmentation and Edge Detection - Approach toRoad Detection in Very High-Resolution Satellite Images

This repository contains the code for the following publication Semantic Segmentation and Edge Detection—Approach to Road Detection in Very High Resolution Satellite Images.
Link: https://www.mdpi.com/2072-4292/14/3/613

### Brief description
This repository is of the network designed to segment roads and road edges in High-resolution remote sensing images. It uses a single hybrid encoder and two decoders for segmentation and the other one for edge detection. Edge detection network uses the combined information from the encoder and the segmentation information. A combination of Weighted cross-entropy and Focal Tversky Loss is used as a loss function to handle the highly imbalanced dataset better.
### Requirements
* Pytorch 1.10.0
* Numpy 1.19.5
* Torchvision 1.11.1

### Usage
* Clone the Repository:
```ruby
  git clone https://github.com/WadiiBoulila/Semantic-Segmentation-Edge-Detection.git
```
* Data:
change the data_path in the `train.py` file.

* Train the model:
```ruby
  python train.py
```

### Dataset Details
Remotesensing image were taken from Riyadah, Jeddah and Dammam and were cropped into 512x512 images. These images were then labeled using VGG Image Annotator. The images and the corresponding labeled mask was then used to train and test the network.

### Model Details
This model uses a hybrid encoder that encodes both high resolution features and low resolution semantic features. The structure of the encoder is shown below:
![alt text](https://www.mdpi.com/remotesensing/remotesensing-14-00613/article_deploy/html/images/remotesensing-14-00613-g002-550.jpg)

The network then uses the encoded features in a cascaded manner to predict both the road segmentation masks and the road edges.

![alt text](https://www.mdpi.com/remotesensing/remotesensing-14-00613/article_deploy/html/images/remotesensing-14-00613-g001-550.jpg)


### Citation

If you use any part of this work please cite using the following Bibtex format:
```
@Article{rs14030613,
AUTHOR = {Ghandorh, Hamza and Boulila, Wadii and Masood, Sharjeel and Koubaa, Anis and Ahmed, Fawad and Ahmad, Jawad},
TITLE = {Semantic Segmentation and Edge Detection&mdash;Approach to Road Detection in Very High Resolution Satellite Images},
JOURNAL = {Remote Sensing},
VOLUME = {14},
YEAR = {2022},
NUMBER = {3},
ARTICLE-NUMBER = {613},
URL = {https://www.mdpi.com/2072-4292/14/3/613},
ISSN = {2072-4292},
DOI = {10.3390/rs14030613}
}

'''
