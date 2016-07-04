# VGG ConvNet for Deeplearning4j

The codes contain the pretrained [Deeplearning4j](http://deeplearning4j.org/) convolutional layers of **ConvNet D (VGG 16)** and **ConvNet E (VGG 19)** described in  

* Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. *arXiv preprint arXiv:1409.1556.*

The pretrained weights were downloaded from [the authors' website](http://www.robots.ox.ac.uk/~vgg/research/very_deep/). The serialized models and configurations in this repository do not contain the fullly connected layers in the original models. The pretrained convolutional networks can be used as feature extractors. 

## Content 
  * `src/Demo.java` for demonstration
    - `INDArray preprocess(INDArray raw)` substracts the means *(R: 123.680, G: 116.779, B: 103.939)* from each channel of input image.
    - `MultiLayerNetwork getBottomLayers(MultiLayerNetwork net, int k)` returns bottom-*k* layers of a pretrained convolutional VGG network.
  * `src/FeatureVisualizer.java`: Visualize features
  * `src/VGGConvNetD.java` and `src/VGGConvNetE.java` contain the configurations of convolutional layers of VGG models.
  * `model/vgg16.dl4jmodel` and `models/vgg19.dl4jmodel` contain serialized Deeplearning4j VGG models
  
## Example 

##### Input: Fat squirrel of U-M  
![squirrel](https://cloud.githubusercontent.com/assets/6327275/16566622/2eecabe4-41e4-11e6-8b3d-03c50e8e8247.jpg)
##### Output of the first 10 layers (7 ConvolutionLayers and 3 SubsamplingLayers) of **VGG ConvNet D**
![vgg16layer10](https://cloud.githubusercontent.com/assets/6327275/16566539/69b50038-41e3-11e6-861f-a1f31fb121f1.png)
##### Final output of **VGG ConvNet E** 
![vgg19layer21](https://cloud.githubusercontent.com/assets/6327275/16566541/6caf96d6-41e3-11e6-868d-b62cb255151f.png)
