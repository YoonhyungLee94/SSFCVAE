# SSFCVAE
This code provides the official PyTorch implementation of the paper "_Boosting Speech Enhancement with Clean Self-Supervised Features via Conditional Variational Autoencoders_", which is submitted to ICASSP 2024.

## Prerequisites
* Ensure you have the VoiceBank-DEMAND dataset. You can download it [here](https://datashare.ed.ac.uk/handle/10283/2791)
* Set up the BigVGAN vocoder. Instructions and the pre-trained model `bigvgan_22khz_80band` can be found [here](https://github.com/NVIDIA/BigVGAN)

## Installation
```bash
# Clone this repository
git clone https://github.com/YoonhyungLee94/SSFCVAE

# Navigate into the directory
cd SSFCVAE

# Install required packages
pip install -r requirements.txt

# Link your dataset and BigVGAN directories
ln -s path_to_your_dataset Dataset
ln -s path_to_bigvgan_repo bigvgan
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
