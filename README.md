# Autoencoders with CNNs

## 1. "beam" dataset

### Training, validation and testing datasets

<img src="./results/loadData_beam_readme.png" width="50%" height="50%" />

### Building and compiling model

<p float="left">
  <img src="./results/buildModel_beam_readme.png" width="50%" height="50%" />
  <img src="./results/compileModel_beam_readme.png" width="50%" height="50%" />
</p>


### Autoencoder training

Training parameters and loss functions obtained durig training.

<p float="left">
  <img src="./results/trainModel_beam_readme.png" width="50%" height="50%" />
  <img src="./results/beam_training_readme.png" width="100%" height="100%" />
</p>

### Prediction results

Results obtained using the autoencoder. 

First row corresponds to the original
image and last row are the recevered image after being passed through
the autoencoder. The middle row corresponds to the latent space (code) representaion, where the
information is compressed preserving the accuracy.

<img src="./results/beam_prediction_readme.png" width="200%" height="200%" />