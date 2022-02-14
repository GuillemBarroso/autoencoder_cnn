# Autoencoders with CNNs

## 1. "mnist" dataset

### Training, validation and testing datasets

<img src="./results/mnist_loadData.png" width="30%" height="30%" />

### Building and compiling model

<p float="left">
  <img src="./results/mnist_buildModel.png" width="30%" height="30%" />
  <img src="./results/mnist_compileModel.png" width="30%" height="30%" />
</p>


### Autoencoder training

Training parameters and loss functions obtained durig training.

<p float="left">
  <img src="./results/mnist_trainModel.png" width="30%" height="30%" />
  <img src="./results/mnist_trainingLoss.png" width="50%" height="50%" />
</p>

### Prediction results

Results obtained using the autoencoder. First row corresponds to the original
image and second row are the recevered image after being passed through
the autoencoder.

<img src="./results/mnist_testResults.png" width="100%" height="100%" />

### Code visulaisation

The code is the condensed information after the input image is passed throguh the 
encoder. In this example the code has a shape (7x7x10) and can be visualised in the 
figure below. Each row corresponds to the 10 filters (columns) of each of the 5 images
shown in the figure above (7, 2, 1, 0 and 4).

<img src="./results/mnist_code.png" width="100%" height="100%" />


## 2. "afreightdata" dataset

### Training, validation and testing datasets

<img src="./images/afreight_loadData.png" width="30%" height="30%" />


### Building and compiling model

<p float="left">
  <img src="./images/afreight_buildModel.png" width="30%" height="30%" />
  <img src="./images/afreight_compileModel.png" width="30%" height="30%" />
</p>

### Autoencoder training

Training parameters and loss functions obtained durig training.

<p float="left">
  <img src="./images/afreight_trainModel.png" width="30%" height="30%" />
  <img src="./images/afreight_trainingLoss.png" width="50%" height="50%" />
</p>

### Prediction results

Results obtained using the autoencoder. First row corresponds to the original
image and second row are the recevered image after being passed through
the autoencoder.

<img src="./images/afreight_testResults.png" width="100%" height="100%" />