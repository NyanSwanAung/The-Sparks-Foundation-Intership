## Task 1 - Object Detection
:dart: Implementing an object detector which identifies the classes of the objects.

<img src="https://raw.githubusercontent.com/NyanSwanAung/The-Sparks-Foundation-Intership/main/assets/Task1.png"/>

In this repo, you'll find codes for 
- Object detection on Image 
- Object detection on Videos (Live webcam in google colab and anaconda environment)

## Object Detection on Image
<a href="https://colab.research.google.com/drive/1OfTtWs5Xty364JAX0lkp918QH32U9OIr?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- :rocket: Model Architecture : Using pre-trained Faster-RCNN and InceptionResNetV2 feature extractor(acting as backbone) to identify objects in images and webcam. InceptionResNetV3 was trained on [ImageNet](http://image-net.org/) and fine-tuned with FasterRCNN on [OpenImages V4 dataset](https://storage.googleapis.com/openimages/web/index.html), containing 600 classes.
This pre-trained model is taken from [TensorFlow Hub](https://www.tensorflow.org/hub).
<img src = "https://raw.githubusercontent.com/NyanSwanAung/The-Sparks-Foundation-Intership/main/assets/faster-rcnn.png"/>

- :bulb: Features : Download image or capture image from webcam and apply object detection.



## :bulb: Steps
1. Import Libraries 
2. Image Preprocessing 
3. Draw bounding box and class name on the image
4. Detector 
5. Take photo from webcam
6. Download pretrained model and predict 

## ✅ Framework and Libraries used 
- TensorFlow 2.0
- Matplotlib
- Numpy
- PIL

## :bookmark: References
- [TensorFlow Hub](https://www.tensorflow.org/hub)
- [Faster-RCNN](https://arxiv.org/abs/1506.01497)
