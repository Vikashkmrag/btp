                                IMAGE SEGMENTATION USING DEEP LEARNING (UNets and RESNets)

# INTRODUCTION

• Convolutional Neural Networks (CNNs) are driving major advances in many computer vision tasks, such as image classification, object detection and semantic image segmentation.

• For image segmentation, we used u-net architecture in which we used resnet34 as encoder and next 34 up sampling layers to give us pixel wise output of an input image.

• We used pretrained resnet-34 model as encoder which was trained on image net dataset.

• So this FCN, with many matrix multiplication, activation functions(relu) pooling layers, up sampling layers, and skip connections gives us robust model which can segment among 12 classes with 94% accuracy.


# About our Data-Set….

The Cambridge-driving Labeled Video Database (CamVid) is the collection of videos with object class semantic labels. The database provides ground truth labels that associate each pixel with one of 32/12 semantic classes. 

• Link to data: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/

# What Model?

• We have used resnet-34 as an encoder in our unettish model.

• Unet and resnet both have skip connection between intermediate hidden layers which adds activation of two layers which helps to regain the features learned by this long network and thus helps in more robust model.

# Why U-Net?

The Advantages of U-Net Architecture are,

The Unet combines the location information from the downsampling path with the contextual information in the upsampling path to finally obtain a general information combining localisation and context, which is necessary to predict a good segmentation map.

No dense layer, so images of different sizes can be used as input (since the only parameters to learn on convolution layers are the kernel, and the size of the kernel is independent from input image’ size).

The use of massive data augmentation is important in domains like segmentation, since the number of annotated samples is usually limited.

# Training

We evaluate our method on CamVid dataset. We trained our models using pretrained resnet-34 without using any extra-data nor post processing module. We evaluated our model with Intersection over Union (IoU) metric and the global accuracy (pixel-wise accuracy on the dataset).

For a given class c, predictions (oi) and targets (yi), the IoU is defined by 

IoU(c) = Pi (oi == c ∧ yi == c) /Pi (oi == c ∨ yi == c)

where ∧ is a logical and operation, while ∨ is a logical or operation. We compute IoU by summing over all the pixels i of the dataset.

We used pretrained model as initialization for down sampling path and used Xavier initialization for rest of our architecture.

Used cross-entropy loss function which has turned to very good in classification task. 

cel=−∑y*log(p) ,where y-actual, p-predicted

rain our model with Adam optimiser, with an initial learning rate of 3e − 3.

We decreased learning rate by factor of 10 as loss function starts decreasing slowly and added exponential weight decay of 0.02 after some time of training so that model won't overfit.

We have trained it over 30 epochs with two different data/image size.

All models are trained on data augmented with random crops and vertical flips, lightning, warping.

We used batch size of 32 as per as our gpu size.

# Results...

• We trained our model over 30 epochs and with two different image size to get accuracy nearly about 94% which is greater than all the previous results In this data set.

• The resulting network is very deep (68 layers). Moreover, it improves t performance on challenging urban scene understanding datasets (CamVid), without neither additional post-processing, nor including temporal information.

• Previous best result on this data was 91.2% using FCN of 103 layer.

• We have implemented our code using python libraries like fastai, pytorch, numpy, PIL etc.

