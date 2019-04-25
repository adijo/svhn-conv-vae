# Convolutional Variational Autoencoder on the SVHN Dataset

## Data Description
* `3 x 32 x 32` tensors.

## Install
```
python3 -m venv venv
source venv/bin/activate
pip install requirements.txt
```

## Train
```
python train.py --gen_images_dir images --num_epochs=100 --batch_size=64
```
Images will be samples and generated at the end of each epoch in the `--gen_images-dir` directory.

