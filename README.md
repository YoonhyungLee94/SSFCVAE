# SSFCVAE
This code provides the official PyTorch implementation of the paper "_Boosting Speech Enhancement with Clean Self-Supervised Features via Conditional Variational Autoencoders_". This work has been submitted to ICASSP 2024.

## Prerequisites
* Ensure you have the VoiceBank-DEMAND dataset. You can download it [here](https://datashare.ed.ac.uk/handle/10283/2791)
* Set up the BigVGAN vocoder. Instructions and the pre-trained model can be found [here](https://github.com/NVIDIA/BigVGAN)
* Move the __init__.py file from the bigvgan_dummy directory to the BigVGAN repo, and complete the remaining steps using the file as a reference

## Installation
```bash
# Clone this repository
git clone https://github.com/YoonhyungLee94/SSFCVAE

# Navigate into the directory
cd SSFCVAE

# Install required packages
pip install -r requirements.txt

# Create a directory for training logs
mkdir training_log

# Set up symbolic links for your dataset and BigVGAN directories
# Replace 'path_to_dataset' with the actual path to the directory containing the VoiceBank-DEMAND dataset folder
# Replace 'path_to_bigvgan_repo' with the actual path to the BigVGAN repository directory
ln -s path_to_dataset Dataset
ln -s path_to_bigvgan_repo bigvgan
mv __init__.py ./bigvgan
```

## Usage
To train the CVAE model, use the following command:

```python
python train.py --gpu 0 --logdir ssf_cvae
```

## Audio samples
Listen to audio samples in the ['./audio_samples'](./audio_samples) directory

## License
This project is licensed under the [MIT License](LICENSE).
