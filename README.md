# Animal10-VGG-Classification
CNN Network based on VGG16 for classification of 10 types of animal images.

I developed a convolutional neural network model for recognising animals from the Kaggle Animal10 database, consisting of approximately 28k medium-quality images. Transfer learning is implemented using VGG16 pre-trained weights for higher efficiency, with a final classification model constructed around this. Hence, this CNN uses 13 convolutional layers, 5 pooling layers, and 2 fully connected layers to classify the images in Animal10 as the appropriate animal. This model achieves % accuracy within 12 epochs, with runtime of approximately one hour using Kaggle GPUs. 

The model was built in Kaggle for easy dataset import and operations, and the original script may be viewed [here](https://www.kaggle.com/marcusdeans/animal10-classification "Kaggle Project"), along with the execution output.

Libraries/frameworks implemented within this project:

* Keras
* TensorFlow
* Numpy
* Pandas
* matplotlib
* scikit-learn
