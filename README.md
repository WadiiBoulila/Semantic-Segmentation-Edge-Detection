# A Semantic Segmentation and Edge Detection - Approach toRoad Detection in Very High-Resolution Satellite Images

### Brief description
This repository is of the network designed to segment roads and road edges in High resolution remote sensing images. It uses a single hybrid encoder and two decoders oen for segmentation and the other one for edge detection. Edge detection network uses the combined information from the encoder and the segmentation information. A combination of Weighted cross-entropy and Focal Tversky Loss is used as a loss function to better handle the highly imbalanced dataset.

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
