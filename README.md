# Autoencoders with CNNs

## Inputs

### Image obtention proess

1. Run FreeFEM code for all 405 cases and store a VTK file for each one.
    * Both homog and SIMP codes.
2. Extract PNGs from the VTKs using Paraview.
    * Paraview exports the entire current view to a specified resolution.
    * The output is then a transparent background with the image (structure) inside.
    * This background has to be removed afterwards for the entire set of images. 
3. Remove background for all PNGs
    * The size (resolution) of the image has been reduced.
    * Is reduced to the size of the structure inside the background box.
    * The size of the structure is given by the Paraview view.
4. Read clean PNGs with the Python library.

### Format

A PNG image can store colour information as:

* RGB: 3 channels with red, green and blue colour magnitudes. A 200x100x3 matrix for instance.
* L: 1 chanell with luminosity (grey scale). A 200x100x1 matrix for instance.
* RGBA: 4 channels with red, green blue and alpha (transparency). A 200x100x4 image.
* LA: 2 channels with luminosity (greyscale) and alpha (transparency).
* P: 1 channel with palette, values from 0 to 256 identifing 256 unique colours. 

This is information is loaded when reading the image. The PILLOW Python library calls it image mode.

### FCNN

Image size:

* A FCNN needs an array as input. Thus, images pixel values are reshaped from 200x100x1 to 20000x1 arrays.
* If an image has more than one channel this grows a lot and the training process time increases a lot.
* An alternative is to reduce it to greyscale. However, if RGBA (4 channels) are converted to L (1 channel), the resulting image contains only a white interface with different tones of black for structure and void.

Image format:

* When importing "beam_homog" dataset there are no issues with the format. All PNGs are RGBA with 4 channels.
* However, with "beam_simp" the PNGs exported form Paraview are both RGBA and P. In fact, only 80/405 are RGBA. 

### CNN

* A CNN can accept a random number of channels. However, they have to be the same for the entire dataset. With "beam_simp" we can only use either RGBA or P. 

# Results

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